#!/usr/bin/env python3
"""
Phase 1a: Core Validation Experiment

This script runs the core validation experiment for Dynamic Context Memory:
- Stage 1: Single-target diagnostic (can we reconstruct Q alone? A alone?)
- Stage 2: Joint Q-A optimization (the core experiment)
- Stage 3: Spot check with greedy decoding

Usage:
    # Run only Stage 2 (default - the core experiment)
    python run_phase1a.py

    # Run specific stage(s)
    python run_phase1a.py --stage 1        # Single-target diagnostic only
    python run_phase1a.py --stage 2,3      # Joint optimization + spot check
    python run_phase1a.py --stage 1,2,3    # All stages

    # Run all stages (shorthand)
    python run_phase1a.py --all

Reference: docs/phase1a-experiment.md
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase1a_optimization import (
    ContentOptimizer,
    OptimizationConfig,
    OptimizationResult,
    SingleTargetOptimizer,
    SingleTargetResult,
    create_mode_embeddings,
    hf_generator_loss_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Force immediate output (no buffering)
logging.getLogger().handlers[0].flush = lambda: sys.stdout.flush()
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
FALLBACK_MODELS = [
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-3B",
    "meta-llama/Llama-3.2-3B",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1a: Core Validation Experiment for DCM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of Q-A pairs to process (default: 100)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=32,
        help="Number of content embedding slots (default: 32)",
    )
    parser.add_argument(
        "--num-restarts",
        type=int,
        default=5,
        help="Number of optimization restarts per example (default: 5)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=800,
        help="Gradient steps per optimization run (default: 800)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--num-reconstructions",
        type=int,
        default=5,
        help="Number of reconstruction examples to log (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cuda', 'cpu', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (default: 0, the 5090). Set to -1 to use all GPUs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files (default: results)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="2",
        help="Stage(s) to run: '1', '2', '3', or comma-separated like '1,2,3' (default: 2)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all stages (equivalent to --stage 1,2,3)",
    )
    parser.add_argument(
        "--stage1-threshold",
        type=float,
        default=0.7,
        help="Minimum convergence rate for Stage 1 to pass (default: 0.7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for decoding: >1.0 discourages repeats (default: 1.0, try 1.2)",
    )
    parser.add_argument(
        "--no-repeat-ngram",
        type=int,
        default=0,
        help="Block repeated n-grams of this size during decoding (default: 0 = disabled, try 2 or 3)",
    )
    # Priority 1: Scheduled Sampling
    parser.add_argument(
        "--scheduled-sampling",
        action="store_true",
        help="Enable scheduled sampling (gradually expose model to its own predictions during training)",
    )
    parser.add_argument(
        "--ss-epsilon-start",
        type=float,
        default=1.0,
        help="Scheduled sampling: initial epsilon (1.0 = pure teacher forcing) (default: 1.0)",
    )
    parser.add_argument(
        "--ss-epsilon-end",
        type=float,
        default=0.2,
        help="Scheduled sampling: final epsilon (0.0 = pure autoregressive) (default: 0.2)",
    )
    parser.add_argument(
        "--ss-decay",
        type=str,
        default="inverse_sigmoid",
        choices=["linear", "exponential", "inverse_sigmoid"],
        help="Scheduled sampling decay schedule (default: inverse_sigmoid)",
    )
    parser.add_argument(
        "--ss-decay-k",
        type=float,
        default=5.0,
        help="Inverse sigmoid steepness: lower=faster decay (default: 5.0, try 3-10)",
    )
    # Priority 2: Unlikelihood Loss
    parser.add_argument(
        "--unlikelihood",
        action="store_true",
        help="Enable unlikelihood training (penalize repetition during training)",
    )
    parser.add_argument(
        "--unlikelihood-alpha",
        type=float,
        default=0.5,
        help="Unlikelihood loss weight (default: 0.5, range: 0.5-1.0)",
    )
    return parser.parse_args()


def parse_stages(args: argparse.Namespace) -> set:
    """Parse which stages to run from args."""
    if args.all:
        return {1, 2, 3}

    stages = set()
    for s in args.stage.split(","):
        s = s.strip()
        if s in ("1", "2", "3"):
            stages.add(int(s))
        else:
            raise ValueError(f"Invalid stage: {s}. Must be 1, 2, or 3.")

    return stages


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_squad_pairs(num_examples: int, seed: int = 42) -> List[Tuple[str, str]]:
    """
    Load Q-A pairs from SQuAD 2.0 dataset.

    Returns:
        List of (question, answer) string pairs
    """
    logger.info(f"Loading SQuAD dataset ({num_examples} examples)...")

    dataset = load_dataset("squad_v2", split="train")

    # Filter to examples with answers (skip impossible questions)
    dataset = dataset.filter(lambda x: len(x["answers"]["text"]) > 0)

    # Shuffle and take subset
    dataset = dataset.shuffle(seed=seed).select(range(min(num_examples, len(dataset))))

    pairs = []
    for example in dataset:
        question = example["question"]
        answer = example["answers"]["text"][0]  # Take first answer
        pairs.append((question, answer))

    logger.info(f"Loaded {len(pairs)} Q-A pairs")
    return pairs


def tokenize_pairs(
    pairs: List[Tuple[str, str]],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Tokenize Q-A pairs.

    Returns:
        List of (Q_tokens, A_tokens) tensor pairs
    """
    tokenized = []
    for q, a in pairs:
        q_tokens = tokenizer.encode(q, return_tensors="pt", add_special_tokens=False).squeeze(0).to(device)
        a_tokens = tokenizer.encode(a, return_tensors="pt", add_special_tokens=False).squeeze(0).to(device)
        tokenized.append((q_tokens, a_tokens))
    return tokenized


# -----------------------------------------------------------------------------
# Baseline Computation
# -----------------------------------------------------------------------------

def compute_baselines(
    model: AutoModelForCausalLM,
    tokenized_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    K: int,
    d: int,
    device: torch.device,
    num_samples: int = 10,
) -> Dict[str, float]:
    """
    Compute baseline losses for calibrating success thresholds.

    Baseline: L(target | random_embeds) - the "no information" baseline.
    Convergence threshold is set relative to this.

    Returns:
        Dict with baseline statistics
    """
    logger.info("Computing baselines...")

    random_losses_q = []
    random_losses_a = []

    # Sample a subset for baseline computation
    sample_pairs = tokenized_pairs[:num_samples]

    with torch.no_grad():
        for q_tokens, a_tokens in tqdm(sample_pairs, desc="Computing baselines"):
            # Random embeddings baseline (float32, converted internally)
            random_embeds = torch.randn(K, d, device=device, dtype=torch.float32) * 0.02

            loss_q = hf_generator_loss_fn(model, q_tokens, random_embeds)
            loss_a = hf_generator_loss_fn(model, a_tokens, random_embeds)

            random_losses_q.append(loss_q.item())
            random_losses_a.append(loss_a.item())

    baselines = {
        "random_q_mean": sum(random_losses_q) / len(random_losses_q),
        "random_a_mean": sum(random_losses_a) / len(random_losses_a),
        "random_combined_mean": (sum(random_losses_q) + sum(random_losses_a)) / (2 * len(random_losses_q)),
    }

    logger.info(f"Baselines computed:")
    logger.info(f"  Random embeddings (Q): {baselines['random_q_mean']:.4f}")
    logger.info(f"  Random embeddings (A): {baselines['random_a_mean']:.4f}")
    logger.info(f"  Random combined: {baselines['random_combined_mean']:.4f}")

    return baselines


# -----------------------------------------------------------------------------
# Stage 1: Single-Target Diagnostic
# -----------------------------------------------------------------------------

def run_stage1(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokenized_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    pairs: List[Tuple[str, str]],
    config: OptimizationConfig,
    device: torch.device,
    num_reconstructions: int = 5,
) -> Tuple[Dict, Dict]:
    """
    Stage 1: Single-target diagnostic.

    Tests whether we can reconstruct Q alone and A alone.
    If this fails, joint optimization won't work either.

    Returns:
        (summary_dict, detailed_results_dict)
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Single-Target Diagnostic")
    logger.info("=" * 60)
    logger.info("Convergence criterion: exact token reconstruction")

    optimizer = SingleTargetOptimizer(
        generator=model,
        generator_loss_fn=hf_generator_loss_fn,
        config=config,
        device=device,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Test on Q targets
    q_targets = [pair[0] for pair in tokenized_pairs]
    logger.info(f"\nOptimizing for Q targets ({len(q_targets)} samples)...")
    q_results = optimizer.run_diagnostic(q_targets)

    # Test on A targets
    a_targets = [pair[1] for pair in tokenized_pairs]
    logger.info(f"\nOptimizing for A targets ({len(a_targets)} samples)...")
    a_results = optimizer.run_diagnostic(a_targets)

    summary = {
        "q_convergence_rate": q_results["convergence_rate"],
        "q_mean_loss": q_results["mean_loss"],
        "a_convergence_rate": a_results["convergence_rate"],
        "a_mean_loss": a_results["mean_loss"],
        "combined_convergence_rate": (q_results["convergence_rate"] + a_results["convergence_rate"]) / 2,
    }

    logger.info(f"\nStage 1 Results (reconstruction-based convergence):")
    logger.info(f"  Q convergence: {summary['q_convergence_rate']:.1%} ({q_results['converged']}/{q_results['total']})")
    logger.info(f"  A convergence: {summary['a_convergence_rate']:.1%} ({a_results['converged']}/{a_results['total']})")
    logger.info(f"  Combined: {summary['combined_convergence_rate']:.1%}")

    # Build per-example detailed results
    per_example_q = []
    per_example_a = []

    for i, r in enumerate(q_results["results"]):
        q_text, _ = pairs[i]
        generated_text = tokenizer.decode(r.generated_tokens, skip_special_tokens=True) if r.generated_tokens else ""
        per_example_q.append({
            "index": i,
            "target_text": q_text,
            "generated_text": generated_text,
            "target_tokens": r.target_tokens,
            "generated_tokens": r.generated_tokens,
            "loss": r.loss,
            "converged": r.converged,
            "steps_to_converge": r.steps_to_converge,
            "num_restarts_tried": r.num_restarts_tried,
        })

    for i, r in enumerate(a_results["results"]):
        _, a_text = pairs[i]
        generated_text = tokenizer.decode(r.generated_tokens, skip_special_tokens=True) if r.generated_tokens else ""
        per_example_a.append({
            "index": i,
            "target_text": a_text,
            "generated_text": generated_text,
            "target_tokens": r.target_tokens,
            "generated_tokens": r.generated_tokens,
            "loss": r.loss,
            "converged": r.converged,
            "steps_to_converge": r.steps_to_converge,
            "num_restarts_tried": r.num_restarts_tried,
        })

    # Log sample reconstructions (all examples, not just converged)
    logger.info(f"\n--- Sample Reconstructions (Stage 1) ---")

    reconstructions_q = []
    reconstructions_a = []

    # Q reconstructions - show first num_reconstructions examples
    for idx in range(min(num_reconstructions, len(q_results["results"]))):
        r = q_results["results"][idx]
        q_text, _ = pairs[idx]
        generated_text = tokenizer.decode(r.generated_tokens, skip_special_tokens=True) if r.generated_tokens else ""

        status = "✓" if r.converged else "✗"
        steps_info = f"steps={r.steps_to_converge}" if r.converged else "no convergence"
        logger.info(f"  {status} Q[{idx}] loss={r.loss:.8f} ({steps_info})")
        logger.info(f"    Target:    {q_text[:80]}{'...' if len(q_text) > 80 else ''}")
        logger.info(f"    Generated: {generated_text[:80]}{'...' if len(generated_text) > 80 else ''}")

        reconstructions_q.append({
            "index": idx,
            "target": q_text,
            "generated": generated_text,
            "loss": r.loss,
            "converged": r.converged,
            "steps_to_converge": r.steps_to_converge,
        })

    # A reconstructions
    for idx in range(min(num_reconstructions, len(a_results["results"]))):
        r = a_results["results"][idx]
        _, a_text = pairs[idx]
        generated_text = tokenizer.decode(r.generated_tokens, skip_special_tokens=True) if r.generated_tokens else ""

        status = "✓" if r.converged else "✗"
        steps_info = f"steps={r.steps_to_converge}" if r.converged else "no convergence"
        logger.info(f"  {status} A[{idx}] loss={r.loss:.8f} ({steps_info})")
        logger.info(f"    Target:    {a_text}")
        logger.info(f"    Generated: {generated_text}")

        reconstructions_a.append({
            "index": idx,
            "target": a_text,
            "generated": generated_text,
            "loss": r.loss,
            "converged": r.converged,
            "steps_to_converge": r.steps_to_converge,
        })

    detailed = {
        "summary": summary,
        "per_example_q": per_example_q,
        "per_example_a": per_example_a,
        "reconstructions_q": reconstructions_q,
        "reconstructions_a": reconstructions_a,
    }

    return summary, detailed


def stage1_passed(results: Dict, threshold: float) -> bool:
    """Check if Stage 1 passed based on convergence threshold."""
    return results["combined_convergence_rate"] >= threshold


# -----------------------------------------------------------------------------
# Stage 2: Joint Q-A Optimization
# -----------------------------------------------------------------------------

def run_stage2(
    model: AutoModelForCausalLM,
    tokenized_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    Q_mode: torch.Tensor,
    A_mode: torch.Tensor,
    config: OptimizationConfig,
    device: torch.device,
    baselines: Dict[str, float],
) -> Tuple[List[OptimizationResult], Dict]:
    """
    Stage 2: Joint Q-A optimization (the core experiment).

    Returns:
        (list of OptimizationResults, summary statistics)
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Joint Q-A Optimization")
    logger.info("=" * 60)

    optimizer = ContentOptimizer(
        generator=model,
        generator_loss_fn=hf_generator_loss_fn,
        Q_mode=Q_mode,
        A_mode=A_mode,
        config=config,
        device=device,
    )

    results = []
    for i, (q_tokens, a_tokens) in enumerate(tqdm(tokenized_pairs, desc="Joint optimization")):
        result = optimizer.optimize(q_tokens, a_tokens)
        results.append(result)

        # Log progress every 20 examples
        if (i + 1) % 20 == 0:
            converged = sum(1 for r in results if r.converged)
            logger.info(f"  Progress: {i+1}/{len(tokenized_pairs)}, converged: {converged}/{i+1}")

    # Compute statistics
    converged_count = sum(1 for r in results if r.converged)
    losses = [r.loss for r in results]
    losses_q = [r.loss_q for r in results]
    losses_a = [r.loss_a for r in results]

    # Compute loss ratios vs random baseline
    random_baseline = baselines["random_combined_mean"]
    loss_ratios = [l / random_baseline for l in losses]

    stats = {
        "total": len(results),
        "converged": converged_count,
        "convergence_rate": converged_count / len(results),
        "mean_loss": sum(losses) / len(losses),
        "mean_loss_q": sum(losses_q) / len(losses_q),
        "mean_loss_a": sum(losses_a) / len(losses_a),
        "min_loss": min(losses),
        "max_loss": max(losses),
        "mean_loss_ratio_vs_random": sum(loss_ratios) / len(loss_ratios),
        "mean_restarts": sum(r.num_restarts_tried for r in results) / len(results),
    }

    logger.info(f"\nStage 2 Results:")
    logger.info(f"  Convergence: {stats['convergence_rate']:.1%} ({converged_count}/{len(results)})")
    logger.info(f"  Mean loss: {stats['mean_loss']:.4f} (Q: {stats['mean_loss_q']:.4f}, A: {stats['mean_loss_a']:.4f})")
    logger.info(f"  Loss ratio vs random: {stats['mean_loss_ratio_vs_random']:.2f}x")
    logger.info(f"  Mean restarts needed: {stats['mean_restarts']:.1f}")

    return results, stats


# -----------------------------------------------------------------------------
# Stage 3: Spot Check with Greedy Decoding
# -----------------------------------------------------------------------------

def run_stage3(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    results: List[OptimizationResult],
    pairs: List[Tuple[str, str]],
    Q_mode: torch.Tensor,
    A_mode: torch.Tensor,
    device: torch.device,
    num_checks: int = 5,
) -> List[Dict]:
    """
    Stage 3: Spot check successful optimizations with greedy decoding.

    For converged examples, decode from (mode, c*) and compare to target.

    Returns:
        List of spot check results
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: Spot Check (Greedy Decoding)")
    logger.info("=" * 60)

    # Get converged examples
    converged_indices = [i for i, r in enumerate(results) if r.converged]
    if not converged_indices:
        logger.warning("No converged examples to spot check!")
        return []

    # Select subset
    check_indices = converged_indices[:num_checks]

    spot_checks = []
    embed_layer = model.get_input_embeddings()

    for idx in check_indices:
        result = results[idx]
        q_text, a_text = pairs[idx]
        content = result.content  # (K, d)

        logger.info(f"\n--- Example {idx} ---")
        logger.info(f"Target Q: {q_text[:100]}...")
        logger.info(f"Target A: {a_text}")

        # Decode from Q_mode + content
        prompt_q = torch.cat([Q_mode, content], dim=0).unsqueeze(0)  # (1, K+1, d)
        generated_q = greedy_decode(model, tokenizer, prompt_q, max_new_tokens=50)
        logger.info(f"Generated Q: {generated_q}")

        # Decode from A_mode + content
        prompt_a = torch.cat([A_mode, content], dim=0).unsqueeze(0)  # (1, K+1, d)
        generated_a = greedy_decode(model, tokenizer, prompt_a, max_new_tokens=30)
        logger.info(f"Generated A: {generated_a}")

        spot_checks.append({
            "index": idx,
            "target_q": q_text,
            "target_a": a_text,
            "generated_q": generated_q,
            "generated_a": generated_a,
            "loss": result.loss,
            "loss_q": result.loss_q,
            "loss_a": result.loss_a,
        })

    return spot_checks


def greedy_decode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_embeds: torch.Tensor,
    max_new_tokens: int = 50,
) -> str:
    """
    Greedy decode from soft prompt embeddings.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        prompt_embeds: Shape (1, prompt_len, d)
        max_new_tokens: Maximum tokens to generate

    Returns:
        Decoded string
    """
    device = prompt_embeds.device
    model_dtype = next(model.parameters()).dtype

    # Convert to model dtype (embeddings may be float32 for optimization stability)
    current_embeds = prompt_embeds.to(dtype=model_dtype)

    generated_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(inputs_embeds=current_embeds)
            next_token_logits = outputs.logits[0, -1, :]

            # Greedy selection
            next_token_id = torch.argmax(next_token_logits).item()
            generated_ids.append(next_token_id)

            # Stop at EOS
            if next_token_id == tokenizer.eos_token_id:
                break

            # Append new token embedding
            next_embed = model.get_input_embeddings()(
                torch.tensor([[next_token_id]], device=device)
            )
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    stages = parse_stages(args)

    logger.info(f"Running stages: {sorted(stages)}")

    # Set seed
    torch.manual_seed(args.seed)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device_map based on --gpu flag
    if device.type == "cuda":
        if args.gpu >= 0:
            device_map = {"": args.gpu}
            logger.info(f"Using GPU {args.gpu} only")
        else:
            device_map = "auto"
            logger.info("Using all available GPUs (device_map=auto)")
    else:
        device_map = None

    # Load model
    logger.info(f"Loading model: {args.model}")
    logger.info("  (this may take a few minutes on first run - downloading weights)")
    try:
        logger.info("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        logger.info("  Tokenizer loaded. Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
        )
        logger.info("  Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load {args.model}: {e}")
        logger.info("Trying fallback models...")
        for fallback in FALLBACK_MODELS:
            try:
                logger.info(f"Trying: {fallback}")
                tokenizer = AutoTokenizer.from_pretrained(fallback, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    fallback,
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    device_map=device_map,
                    trust_remote_code=True,
                )
                logger.info(f"  Fallback model {fallback} loaded successfully.")
                args.model = fallback
                break
            except Exception:
                continue
        else:
            logger.error("All model loading attempts failed!")
            sys.exit(1)

    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Get model dimension
    d = model.config.hidden_size
    logger.info(f"Model hidden size: {d}")

    # Create mode embeddings using "Q" and "A" tokens
    # Try to find reasonable tokens for mode selection
    try:
        q_token_id = tokenizer.encode("Q", add_special_tokens=False)[0]
        a_token_id = tokenizer.encode("A", add_special_tokens=False)[0]
    except (IndexError, KeyError):
        # Fallback to arbitrary distinct tokens
        q_token_id = 81   # Often 'Q' in many tokenizers
        a_token_id = 65   # Often 'A' in many tokenizers

    Q_mode, A_mode = create_mode_embeddings(model, q_token_id, a_token_id)
    Q_mode = Q_mode.to(device)
    A_mode = A_mode.to(device)
    logger.info(f"Created mode embeddings (Q: token {q_token_id}, A: token {a_token_id})")

    # Load dataset
    pairs = load_squad_pairs(args.num_examples, seed=args.seed)
    tokenized_pairs = tokenize_pairs(pairs, tokenizer, device)

    # Create optimization config
    config = OptimizationConfig(
        K=args.K,
        d=d,
        num_restarts=args.num_restarts,
        num_steps=args.num_steps,
        lr=args.lr,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
        # Scheduled sampling (Priority 1)
        use_scheduled_sampling=args.scheduled_sampling,
        ss_initial_epsilon=args.ss_epsilon_start,
        ss_final_epsilon=args.ss_epsilon_end,
        ss_decay_type=args.ss_decay,
        ss_decay_k=args.ss_decay_k,
        # Unlikelihood (Priority 2)
        use_unlikelihood=args.unlikelihood,
        unlikelihood_alpha=args.unlikelihood_alpha,
    )
    logger.info(f"Optimization config: K={config.K}, steps={config.num_steps}, restarts={config.num_restarts}")
    if args.repetition_penalty != 1.0 or args.no_repeat_ngram > 0:
        logger.info(f"  Inference anti-repetition: penalty={args.repetition_penalty}, no_repeat_ngram={args.no_repeat_ngram}")
    if args.scheduled_sampling:
        logger.info(f"  Scheduled sampling: ε {args.ss_epsilon_start}→{args.ss_epsilon_end} ({args.ss_decay})")
    if args.unlikelihood:
        logger.info(f"  Unlikelihood training: α={args.unlikelihood_alpha}")

    # Compute baselines
    baselines = compute_baselines(model, tokenized_pairs, args.K, d, device)

    # Update loss threshold based on baselines
    # Convergence = loss < 0.01 * random_baseline (very conservative - we've seen 0.001 achieved)
    config.loss_threshold = min(0.1, baselines["random_combined_mean"] * 0.01)
    logger.info(f"Dynamic loss threshold: {config.loss_threshold:.4f} (min of 0.1 or 0.01x random baseline)")

    # Stage 1: Single-target diagnostic
    stage1_summary = None
    stage1_detailed = None
    if 1 in stages:
        stage1_summary, stage1_detailed = run_stage1(
            model, tokenizer, tokenized_pairs, pairs, config, device,
            num_reconstructions=args.num_reconstructions,
        )

        if not stage1_passed(stage1_summary, args.stage1_threshold):
            logger.warning(f"Stage 1 FAILED: convergence {stage1_summary['combined_convergence_rate']:.1%} < {args.stage1_threshold:.1%}")
            logger.warning("Joint optimization unlikely to succeed. Consider:")
            logger.warning("  - Increasing K")
            logger.warning("  - Adjusting learning rate")
            logger.warning("  - Trying a different model")
        else:
            logger.info(f"Stage 1 PASSED: convergence {stage1_summary['combined_convergence_rate']:.1%}")

    # Stage 2: Joint Q-A optimization
    stage2_results = None
    stage2_stats = None
    if 2 in stages:
        stage2_results, stage2_stats = run_stage2(
            model, tokenized_pairs, Q_mode, A_mode, config, device, baselines
        )

    # Stage 3: Spot check (requires stage 2 results)
    spot_checks = []
    if 3 in stages:
        if stage2_results is None:
            logger.warning("Stage 3 requires Stage 2 results. Running Stage 2 first...")
            stage2_results, stage2_stats = run_stage2(
                model, tokenized_pairs, Q_mode, A_mode, config, device, baselines
            )
        spot_checks = run_stage3(
            model, tokenizer, stage2_results, pairs, Q_mode, A_mode, device
        )

    # Summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Examples: {args.num_examples}, K: {args.K}")
    logger.info(f"Stages run: {sorted(stages)}")

    if stage1_summary:
        logger.info(f"Stage 1 convergence: {stage1_summary['combined_convergence_rate']:.1%}")

    if stage2_stats:
        logger.info(f"Stage 2 convergence: {stage2_stats['convergence_rate']:.1%}")
        logger.info(f"Mean loss ratio vs random: {stage2_stats['mean_loss_ratio_vs_random']:.2f}x")

        # Assess success
        if stage2_stats["convergence_rate"] >= 0.5:
            logger.info("RESULT: Core validation PASSED (>50% convergence)")
        elif stage2_stats["convergence_rate"] >= 0.3:
            logger.info("RESULT: Marginal (30-50% convergence) - investigate hyperparameters")
        else:
            logger.info("RESULT: Core validation FAILED (<30% convergence)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"phase1a_results_{timestamp}.json"

    # Prepare serializable results
    serializable_results = {
        "config": {
            "model": args.model,
            "num_examples": args.num_examples,
            "K": args.K,
            "num_restarts": args.num_restarts,
            "num_steps": args.num_steps,
            "lr": args.lr,
            "d": d,
            "loss_threshold": config.loss_threshold,
            "repetition_penalty": config.repetition_penalty,
            "no_repeat_ngram_size": config.no_repeat_ngram_size,
            "use_scheduled_sampling": config.use_scheduled_sampling,
            "ss_initial_epsilon": config.ss_initial_epsilon,
            "ss_final_epsilon": config.ss_final_epsilon,
            "ss_decay_type": config.ss_decay_type,
            "use_unlikelihood": config.use_unlikelihood,
            "unlikelihood_alpha": config.unlikelihood_alpha,
            "stages_run": sorted(stages),
        },
        "baselines": baselines,
        "stage1": stage1_detailed if stage1_detailed else stage1_summary,
        "stage2": stage2_stats,
        "stage2_per_example": [
            {
                "loss": r.loss,
                "loss_q": r.loss_q,
                "loss_a": r.loss_a,
                "converged": r.converged,
                "num_restarts_tried": r.num_restarts_tried,
            }
            for r in stage2_results
        ] if stage2_results else None,
        "spot_checks": spot_checks if spot_checks else None,
        "timestamp": timestamp,
    }

    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")

    # Also save the optimized embeddings for successful cases
    if stage2_results:
        converged_embeddings = [
            (i, r.content.cpu())
            for i, r in enumerate(stage2_results)
            if r.converged
        ]
        if converged_embeddings:
            embeddings_file = output_dir / f"phase1a_embeddings_{timestamp}.pt"
            torch.save(
                {
                    "embeddings": {i: emb for i, emb in converged_embeddings},
                    "Q_mode": Q_mode.cpu(),
                    "A_mode": A_mode.cpu(),
                    "config": asdict(config),
                },
                embeddings_file,
            )
            logger.info(f"Embeddings saved to: {embeddings_file}")


if __name__ == "__main__":
    main()
