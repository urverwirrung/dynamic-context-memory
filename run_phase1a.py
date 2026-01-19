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
import platform
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional experiment tracking
try:
    from experiment_tracker import get_tracker, generate_run_id
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False

# Optional experiment-runner library for distributed execution
try:
    from experiment_runner import (
        DistributedExecutor,
        FilesystemBackend,
        RetryPolicy,
        SerialExecutor,
        WorkerSpec,
    )
    from phase1a_task import Phase1aTask, Phase1aInput, create_inputs_from_pairs
    EXPERIMENT_RUNNER_AVAILABLE = True
except ImportError:
    EXPERIMENT_RUNNER_AVAILABLE = False

from phase1a_optimization import (
    ContentOptimizer,
    OptimizationConfig,
    OptimizationResult,
    SingleTargetOptimizer,
    SingleTargetResult,
    create_mode_embeddings,
    hf_generator_loss_fn,
    hf_semantic_loss_fn,
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
        "--lr-warm-restarts",
        action="store_true",
        help="Use cosine annealing with warm restarts (LR periodically jumps back up)",
    )
    parser.add_argument(
        "--lr-restart-t0",
        type=int,
        default=400,
        help="Initial restart period for warm restarts (default: 400)",
    )
    parser.add_argument(
        "--lr-restart-mult",
        type=int,
        default=2,
        help="Restart period multiplier (default: 2, so 400 -> 800 -> 1600...)",
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
        "--parallel",
        action="store_true",
        help="Run Stage 2 in parallel on GPU 0 and GPU 1 (doubles throughput)",
    )
    parser.add_argument(
        "--distributed",
        type=str,
        default=None,
        help="Run distributed across machines. Format: 'host:gpu:weight,...' e.g. 'localhost:0:100,localhost:1:70,miranda:0:70,miranda:1:55'. Weights are relative compute (5090=100, 5080=70, 5070Ti=55). Weight is optional (defaults to 100).",
    )
    parser.add_argument(
        "--remote-dir",
        type=str,
        default="~/repos/dynamic-context-memory",
        help="Path to code on remote machines (default: ~/repos/dynamic-context-memory)",
    )
    parser.add_argument(
        "--remote-venv",
        type=str,
        default="~/.venv/dcm",
        help="Path to virtualenv on remote machines (default: ~/.venv/dcm)",
    )
    parser.add_argument(
        "--shared-fs",
        type=str,
        default="/tmp/experiment-runner",
        help="Path to shared filesystem for distributed execution (default: /tmp/experiment-runner). Required for experiment-runner mode.",
    )
    parser.add_argument(
        "--use-experiment-runner",
        action="store_true",
        help="Use experiment-runner library for distributed execution (requires pip install -e ~/repos/experiment-runner)",
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
        "--add-eos",
        action="store_true",
        default=True,
        help="Add EOS token to targets (teaches model to stop) (default: enabled)",
    )
    parser.add_argument(
        "--no-eos",
        action="store_false",
        dest="add_eos",
        help="Don't add EOS token to targets",
    )
    parser.add_argument(
        "--eos-convergence",
        action="store_true",
        default=True,
        help="Use P(EOS) threshold for convergence (default: enabled)",
    )
    parser.add_argument(
        "--no-eos-convergence",
        action="store_false",
        dest="eos_convergence",
        help="Disable EOS-based convergence (require exact match)",
    )
    parser.add_argument(
        "--eos-threshold",
        type=float,
        default=0.5,
        help="P(EOS) threshold for convergence (default: 0.5)",
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
    parser.add_argument(
        "--ss-min-epsilon",
        type=float,
        default=0.1,
        help="Minimum epsilon floor (prevents collapse to pure autoregressive) (default: 0.1)",
    )
    parser.add_argument(
        "--ss-nudge-up",
        action="store_true",
        default=True,
        help="Enable nudging epsilon UP when degeneration detected (default: enabled)",
    )
    parser.add_argument(
        "--no-ss-nudge-up",
        action="store_false",
        dest="ss_nudge_up",
        help="Disable nudging epsilon UP (only nudge down)",
    )
    parser.add_argument(
        "--ss-nudge-up-factor",
        type=float,
        default=1.5,
        help="Factor to multiply epsilon when nudging UP (default: 1.5)",
    )
    parser.add_argument(
        "--ss-nudge-up-max",
        type=float,
        default=0.8,
        help="Maximum epsilon when nudging UP (default: 0.8)",
    )
    # Semantic loss: reward semantic similarity instead of exact token match
    parser.add_argument(
        "--semantic-loss",
        action="store_true",
        help="Use semantic similarity loss (partial credit for similar tokens like 'London' vs 'Greater London')",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=1.0,
        help="Weight for semantic loss (1.0 = pure semantic, 0.5 = blend with CE) (default: 1.0)",
    )
    parser.add_argument(
        "--semantic-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for semantic loss (lower = sharper) (default: 1.0)",
    )
    parser.add_argument(
        "--semantic-sentence-weight",
        type=float,
        default=0.5,
        help="Weight for sentence-level vs token-level loss (0 = token only, 1 = sentence only) (default: 0.5)",
    )
    parser.add_argument(
        "--eos-penalty-weight",
        type=float,
        default=1.0,
        help="Weight for extra token penalty (penalizes non-EOS after target) (default: 1.0)",
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
    # Experiment tracking (requires experiment-tracker package)
    # Tracking is enabled by default if experiment-tracker is installed
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable experiment tracking",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="dcm",
        help="Project name for tracking (default: dcm)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="phase1a",
        help="Experiment name for tracking (default: phase1a)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Tags for experiment tracking",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Notes for experiment tracking",
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


def get_git_commit() -> Optional[str]:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def get_hostname() -> str:
    """Get current machine hostname."""
    return socket.gethostname().split(".")[0].lower()


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
    add_eos: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Tokenize Q-A pairs.

    Args:
        pairs: List of (question, answer) string pairs
        tokenizer: HuggingFace tokenizer
        device: Target device for tensors
        add_eos: If True, append EOS token to each target (teaches model to stop)

    Returns:
        List of (Q_tokens, A_tokens) tensor pairs
    """
    tokenized = []
    eos_token_id = tokenizer.eos_token_id
    for q, a in pairs:
        q_tokens = tokenizer.encode(q, return_tensors="pt", add_special_tokens=False).squeeze(0).to(device)
        a_tokens = tokenizer.encode(a, return_tensors="pt", add_special_tokens=False).squeeze(0).to(device)

        # Append EOS to teach model when to stop
        if add_eos and eos_token_id is not None:
            eos_tensor = torch.tensor([eos_token_id], device=device)
            q_tokens = torch.cat([q_tokens, eos_tensor])
            a_tokens = torch.cat([a_tokens, eos_tensor])

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
    use_semantic_loss: bool = False,
    semantic_sentence_weight: float = 0.5,
    eos_token_id: Optional[int] = None,
    eos_penalty_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Compute baseline losses for calibrating success thresholds.

    Baseline: L(target | random_embeds) - the "no information" baseline.
    Convergence threshold is set relative to this.

    Args:
        use_semantic_loss: If True, compute baselines with semantic loss instead of CE
        semantic_sentence_weight: Weight for sentence-level loss (only used with semantic loss)

    Returns:
        Dict with baseline statistics
    """
    loss_type = "semantic" if use_semantic_loss else "CE"
    logger.info(f"Computing baselines (using {loss_type} loss)...")

    random_losses_q = []
    random_losses_a = []

    # Sample a subset for baseline computation
    sample_pairs = tokenized_pairs[:num_samples]

    with torch.no_grad():
        for q_tokens, a_tokens in tqdm(sample_pairs, desc="Computing baselines"):
            # Random embeddings baseline (float32, converted internally)
            random_embeds = torch.randn(K, d, device=device, dtype=torch.float32) * 0.02

            if use_semantic_loss:
                loss_q, _, _ = hf_semantic_loss_fn(
                    model, q_tokens, random_embeds,
                    sentence_weight=semantic_sentence_weight,
                    eos_token_id=eos_token_id,
                    eos_penalty_weight=eos_penalty_weight,
                )
                loss_a, _, _ = hf_semantic_loss_fn(
                    model, a_tokens, random_embeds,
                    sentence_weight=semantic_sentence_weight,
                    eos_token_id=eos_token_id,
                    eos_penalty_weight=eos_penalty_weight,
                )
            else:
                loss_q = hf_generator_loss_fn(model, q_tokens, random_embeds)
                loss_a = hf_generator_loss_fn(model, a_tokens, random_embeds)

            random_losses_q.append(loss_q.item())
            random_losses_a.append(loss_a.item())

    baselines = {
        "random_q_mean": sum(random_losses_q) / len(random_losses_q),
        "random_a_mean": sum(random_losses_a) / len(random_losses_a),
        "random_combined_mean": (sum(random_losses_q) + sum(random_losses_a)) / (2 * len(random_losses_q)),
        "loss_type": loss_type,
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
        tokenizer=tokenizer,
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
    tokenizer: AutoTokenizer,
    tokenized_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    pairs: List[Tuple[str, str]],
    Q_mode: torch.Tensor,
    A_mode: torch.Tensor,
    config: OptimizationConfig,
    device: torch.device,
    baselines: Dict[str, float],
) -> Tuple[List[OptimizationResult], Dict]:
    """
    Stage 2: Joint Q-A optimization (the core experiment).

    Convergence = BOTH Q and A reconstruct correctly from a single soft prompt.

    Returns:
        (list of OptimizationResults, summary statistics)
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Joint Q-A Optimization")
    logger.info("=" * 60)
    logger.info("Convergence criterion: both Q and A reconstruct exactly")

    optimizer = ContentOptimizer(
        generator=model,
        generator_loss_fn=hf_generator_loss_fn,
        Q_mode=Q_mode,
        A_mode=A_mode,
        config=config,
        device=device,
        eos_token_id=tokenizer.eos_token_id,
        tokenizer=tokenizer,
    )

    results = []
    num_examples = len(tokenized_pairs)

    # Single progress bar over all steps across all examples and restarts
    total_steps = num_examples * config.num_restarts * config.num_steps
    with tqdm(total=total_steps, desc="Joint optimization", dynamic_ncols=True) as pbar:
        for i, (q_tokens, a_tokens) in enumerate(tokenized_pairs):
            result = optimizer.optimize(
                q_tokens, a_tokens,
                pbar=pbar, example_idx=i, num_examples=num_examples,
            )
            results.append(result)

            # Log each result
            status = "✓" if result.converged else "✗"
            q_text, a_text = pairs[i]
            tqdm.write(f"  {status} [{i}] loss={result.loss:.6f} (Q: {result.loss_q:.6f}, A: {result.loss_a:.6f}) restarts: {result.num_restarts_tried}")
            if not result.converged:
                tqdm.write(f"      Q: {q_text[:60]}{'...' if len(q_text) > 60 else ''}")
                tqdm.write(f"      A: {a_text[:60]}{'...' if len(a_text) > 60 else ''}")

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


def _stage2_worker(
    gpu_id: int,
    model_name: str,
    pairs_subset: List[Tuple[str, str]],
    indices: List[int],
    config_dict: Dict,
    add_eos: bool,
    worker_id: int = 0,
) -> List[Tuple[int, Dict]]:
    """
    Worker function for parallel Stage 2 execution.

    Loads model on specified GPU and processes a subset of examples.
    Returns list of (original_index, result_dict) tuples.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from phase1a_optimization import (
        ContentOptimizer, OptimizationConfig, create_mode_embeddings,
        hf_generator_loss_fn,
    )
    from tqdm import tqdm

    device = torch.device(f"cuda:{gpu_id}")

    # Load model on this GPU
    print(f"[Worker {worker_id} / GPU {gpu_id}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": gpu_id},
        trust_remote_code=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    d = model.config.hidden_size

    # Create mode embeddings
    q_token_id = tokenizer.encode("Q", add_special_tokens=False)[0]
    a_token_id = tokenizer.encode("A", add_special_tokens=False)[0]
    Q_mode, A_mode = create_mode_embeddings(model, q_token_id, a_token_id)
    Q_mode = Q_mode.to(device)
    A_mode = A_mode.to(device)

    # Tokenize pairs
    tokenized = []
    eos_token_id = tokenizer.eos_token_id
    for q, a in pairs_subset:
        q_tokens = tokenizer.encode(q, return_tensors="pt", add_special_tokens=False).squeeze(0).to(device)
        a_tokens = tokenizer.encode(a, return_tensors="pt", add_special_tokens=False).squeeze(0).to(device)
        if add_eos and eos_token_id is not None:
            eos_tensor = torch.tensor([eos_token_id], device=device)
            q_tokens = torch.cat([q_tokens, eos_tensor])
            a_tokens = torch.cat([a_tokens, eos_tensor])
        tokenized.append((q_tokens, a_tokens))

    # Create config
    config = OptimizationConfig(**config_dict)

    # Create optimizer
    optimizer = ContentOptimizer(
        generator=model,
        generator_loss_fn=hf_generator_loss_fn,
        Q_mode=Q_mode,
        A_mode=A_mode,
        config=config,
        device=device,
        eos_token_id=tokenizer.eos_token_id,
        tokenizer=tokenizer,
    )

    # Run optimization
    results = []
    num_examples = len(tokenized)
    total_steps = num_examples * config.num_restarts * config.num_steps

    print(f"[Worker {worker_id} / GPU {gpu_id}] Processing {num_examples} examples...")
    with tqdm(total=total_steps, desc=f"W{worker_id}/GPU{gpu_id}", position=worker_id, dynamic_ncols=True) as pbar:
        for i, (q_tokens, a_tokens) in enumerate(tokenized):
            result = optimizer.optimize(
                q_tokens, a_tokens,
                pbar=pbar, example_idx=i, num_examples=num_examples,
            )
            original_idx = indices[i]
            results.append((original_idx, {
                "loss": result.loss,
                "loss_q": result.loss_q,
                "loss_a": result.loss_a,
                "converged": result.converged,
                "num_restarts_tried": result.num_restarts_tried,
                "steps_to_converge": result.steps_to_converge,
            }))

            status = "✓" if result.converged else "✗"
            q_text, a_text = pairs_subset[i]
            tqdm.write(f"  {status} [{original_idx}] W{worker_id}/GPU{gpu_id} loss={result.loss:.6f}")

    print(f"[Worker {worker_id} / GPU {gpu_id}] Done. Converged: {sum(1 for _, r in results if r['converged'])}/{len(results)}")
    return results


def run_stage2_parallel(
    model_name: str,
    pairs: List[Tuple[str, str]],
    config_dict: Dict,
    add_eos: bool,
    baselines: Dict[str, float],
    num_workers: int = 2,
) -> Tuple[List[Dict], Dict]:
    """
    Run Stage 2 in parallel across GPUs.

    Default: 2 workers (1 per GPU). Even though GPU 0 has enough VRAM for 2 models,
    the GPU is compute-bound so multiple workers on same GPU causes contention.

    Returns:
        (list of result dicts in original order, summary statistics)
    """
    logger.info("=" * 60)
    logger.info(f"STAGE 2: Joint Q-A Optimization (PARALLEL - {num_workers} workers)")
    logger.info("=" * 60)
    logger.info("Convergence criterion: both Q and A reconstruct exactly")

    # Split examples between workers (1 worker per GPU)
    n = len(pairs)
    chunk_size = n // num_workers
    remainder = n % num_workers

    worker_assignments = []
    start = 0
    for i in range(num_workers):
        # Distribute remainder among first workers
        end = start + chunk_size + (1 if i < remainder else 0)
        # Simple 1:1 mapping: worker i -> GPU i
        gpu_id = i
        worker_assignments.append({
            "worker_id": i,
            "gpu_id": gpu_id,
            "pairs": pairs[start:end],
            "indices": list(range(start, end)),
        })
        logger.info(f"  Worker {i} (GPU {gpu_id}): examples {start}-{end-1} ({end-start} examples)")
        start = end

    # Use spawn to avoid CUDA issues with fork
    ctx = mp.get_context('spawn')

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = []
        for w in worker_assignments:
            future = executor.submit(
                _stage2_worker, w["gpu_id"], model_name, w["pairs"], w["indices"], config_dict, add_eos, w["worker_id"]
            )
            futures.append(future)

        # Collect results
        all_results = []
        for future in futures:
            all_results.extend(future.result())

    # Merge results in original order
    all_results.sort(key=lambda x: x[0])  # Sort by original index
    results = [r for _, r in all_results]

    # Compute statistics
    converged_count = sum(1 for r in results if r["converged"])
    losses = [r["loss"] for r in results]
    losses_q = [r["loss_q"] for r in results]
    losses_a = [r["loss_a"] for r in results]

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
        "mean_restarts": sum(r["num_restarts_tried"] for r in results) / len(results),
    }

    logger.info(f"\nStage 2 Results (Parallel - {num_workers} workers):")
    logger.info(f"  Convergence: {stats['convergence_rate']:.1%} ({converged_count}/{len(results)})")
    logger.info(f"  Mean loss: {stats['mean_loss']:.4f} (Q: {stats['mean_loss_q']:.4f}, A: {stats['mean_loss_a']:.4f})")
    logger.info(f"  Loss ratio vs random: {stats['mean_loss_ratio_vs_random']:.2f}x")
    logger.info(f"  Mean restarts needed: {stats['mean_restarts']:.1f}")

    return results, stats


def _run_worker_subprocess(
    host: str,
    gpu_id: int,
    worker_id: int,
    model_name: str,
    pairs: List[Tuple[str, str]],
    indices: List[int],
    config_dict: Dict,
    add_eos: bool,
    remote_dir: str,
    remote_venv: str = None,
) -> List[Dict]:
    """
    Run a worker via subprocess (local) or SSH (remote).

    Returns list of result dicts.
    """
    import subprocess
    import json

    work_config = {
        "gpu_id": gpu_id,
        "model_name": model_name,
        "pairs": pairs,
        "indices": indices,
        "config": config_dict,
        "add_eos": add_eos,
        "worker_id": worker_id,
    }
    input_json = json.dumps(work_config)

    if host == "localhost":
        # Local subprocess
        cmd = ["python3", "phase1a_worker.py"]
        logger.info(f"  Starting local worker {worker_id} on GPU {gpu_id}...")
    else:
        # Remote via SSH - activate venv if specified
        if remote_venv:
            ssh_cmd = f"source {remote_venv}/bin/activate && cd {remote_dir} && python3 phase1a_worker.py"
        else:
            ssh_cmd = f"cd {remote_dir} && python3 phase1a_worker.py"
        cmd = ["ssh", host, ssh_cmd]
        logger.info(f"  Starting remote worker {worker_id} on {host}:GPU{gpu_id}...")
        logger.debug(f"  SSH command: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Send input and close stdin
    proc.stdin.write(input_json)
    proc.stdin.close()

    # Stream stderr in real-time while collecting stdout
    import threading
    import io

    stderr_lines = []
    def stream_stderr():
        for line in proc.stderr:
            line = line.rstrip()
            stderr_lines.append(line)
            print(f"[W{worker_id}] {line}", file=sys.stderr, flush=True)

    stderr_thread = threading.Thread(target=stream_stderr)
    stderr_thread.start()

    # Collect stdout
    stdout = proc.stdout.read()
    proc.wait()
    stderr_thread.join()

    stderr = '\n'.join(stderr_lines)

    if proc.returncode != 0:
        logger.error(f"Worker {worker_id} failed with return code {proc.returncode}")
        logger.error(f"stderr: {stderr}")
        return []

    # Parse results from stdout
    try:
        results = json.loads(stdout)
        return results
    except json.JSONDecodeError as e:
        logger.error(f"Worker {worker_id} returned invalid JSON: {e}")
        logger.error(f"stdout: {stdout[:500]}")
        return []


def run_stage2_distributed(
    worker_specs: str,
    model_name: str,
    pairs: List[Tuple[str, str]],
    config_dict: Dict,
    add_eos: bool,
    baselines: Dict[str, float],
    remote_dir: str,
    remote_venv: str = None,
) -> Tuple[List[Dict], Dict]:
    """
    Run Stage 2 distributed across multiple machines/GPUs.

    worker_specs format: "host:gpu:weight,host:gpu:weight,..."
    Example: "localhost:0:100,localhost:1:70,miranda:0:70,miranda:1:55"

    Weights are relative compute throughput (higher = faster = more examples).
    Suggested weights based on GPU:
      - RTX 5090: 100
      - RTX 5080: 70
      - RTX 5070 Ti: 55

    Returns:
        (list of result dicts in original order, summary statistics)
    """
    from concurrent.futures import ThreadPoolExecutor

    logger.info("=" * 60)
    logger.info("STAGE 2: Joint Q-A Optimization (DISTRIBUTED)")
    logger.info("=" * 60)
    logger.info("Convergence criterion: both Q and A reconstruct exactly")

    # Parse worker specs
    workers = []
    for i, spec in enumerate(worker_specs.split(",")):
        parts = spec.strip().split(":")
        if len(parts) == 2:
            host, gpu = parts
            weight = 100  # default weight
        elif len(parts) == 3:
            host, gpu, weight = parts
            weight = int(weight)
        else:
            raise ValueError(f"Invalid worker spec: {spec}. Expected host:gpu or host:gpu:weight")
        workers.append({
            "worker_id": i,
            "host": host,
            "gpu_id": int(gpu),
            "weight": weight,
        })

    # Distribute examples proportionally to weights
    total_weight = sum(w["weight"] for w in workers)
    n = len(pairs)

    start = 0
    for i, w in enumerate(workers):
        # Calculate proportional share (last worker gets remainder)
        if i == len(workers) - 1:
            count = n - start
        else:
            count = int(n * w["weight"] / total_weight)
        end = start + count
        w["pairs"] = pairs[start:end]
        w["indices"] = list(range(start, end))
        logger.info(f"  Worker {i} ({w['host']}:GPU{w['gpu_id']}, weight={w['weight']}): examples {start}-{end-1} ({count} examples)")
        start = end

    # Run all workers in parallel using threads (each spawns subprocess/SSH)
    all_results = []

    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = []
        for w in workers:
            future = executor.submit(
                _run_worker_subprocess,
                w["host"], w["gpu_id"], w["worker_id"],
                model_name, w["pairs"], w["indices"],
                config_dict, add_eos, remote_dir, remote_venv,
            )
            futures.append(future)

        # Collect results
        for future in futures:
            results = future.result()
            all_results.extend(results)

    # Sort by original index
    all_results.sort(key=lambda x: x["index"])
    results = [{k: v for k, v in r.items() if k != "index"} for r in all_results]

    # Compute statistics
    converged_count = sum(1 for r in results if r["converged"])
    losses = [r["loss"] for r in results]
    losses_q = [r["loss_q"] for r in results]
    losses_a = [r["loss_a"] for r in results]

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
        "mean_restarts": sum(r["num_restarts_tried"] for r in results) / len(results),
    }

    logger.info(f"\nStage 2 Results (Distributed - {len(workers)} workers):")
    logger.info(f"  Convergence: {stats['convergence_rate']:.1%} ({converged_count}/{len(results)})")
    logger.info(f"  Mean loss: {stats['mean_loss']:.4f} (Q: {stats['mean_loss_q']:.4f}, A: {stats['mean_loss_a']:.4f})")
    logger.info(f"  Loss ratio vs random: {stats['mean_loss_ratio_vs_random']:.2f}x")
    logger.info(f"  Mean restarts needed: {stats['mean_restarts']:.1f}")

    return results, stats


def run_stage2_experiment_runner(
    worker_specs: str,
    model_name: str,
    pairs: List[Tuple[str, str]],
    config_dict: Dict,
    add_eos: bool,
    baselines: Dict[str, float],
    shared_fs_path: str,
    remote_dir: str,
    remote_venv: str = None,
) -> Tuple[List[Dict], Dict]:
    """
    Run Stage 2 using the experiment-runner library.

    Uses the new distributed execution framework with proper artifact handling,
    retry policies, and tmux-based workers.

    Args:
        worker_specs: "host:gpu:weight,..." format
        model_name: HuggingFace model name
        pairs: List of (Q, A) string pairs
        config_dict: OptimizationConfig parameters
        add_eos: Whether to append EOS to targets
        baselines: Baseline loss values
        shared_fs_path: Path to shared filesystem for artifacts
        remote_dir: Working directory on remote machines
        remote_venv: Path to virtualenv on remote machines

    Returns:
        (list of result dicts, summary statistics)
    """
    if not EXPERIMENT_RUNNER_AVAILABLE:
        logger.error("experiment-runner library not installed!")
        logger.error("Install with: pip install -e ~/repos/experiment-runner")
        raise ImportError("experiment-runner library required for this mode")

    logger.info("=" * 60)
    logger.info("STAGE 2: Joint Q-A Optimization (EXPERIMENT-RUNNER)")
    logger.info("=" * 60)
    logger.info("Convergence criterion: both Q and A reconstruct exactly")

    # Parse worker specs
    workers = []
    for spec in worker_specs.split(","):
        parts = spec.strip().split(":")
        if len(parts) == 2:
            host, gpu = parts
            weight = 100
        elif len(parts) == 3:
            host, gpu, weight = parts
            weight = int(weight)
        else:
            raise ValueError(f"Invalid worker spec: {spec}")

        workers.append(WorkerSpec(
            host=host,
            gpu_id=int(gpu),
            weight=weight,
            python_path=f"{remote_venv}/bin/python" if remote_venv else "python",
            work_dir=remote_dir,
        ))

    logger.info(f"  Workers: {len(workers)}")
    for w in workers:
        logger.info(f"    - {w.host}:GPU{w.gpu_id} (weight={w.weight})")

    # Set up storage backend
    storage = FilesystemBackend(Path(shared_fs_path))
    if not storage.verify_local_access():
        raise RuntimeError(f"Cannot access shared filesystem: {shared_fs_path}")

    # Create task
    task = Phase1aTask(
        model_name=model_name,
        config_dict=config_dict,
        add_eos=add_eos,
        save_embeddings=True,
    )

    # Create inputs
    inputs = create_inputs_from_pairs(pairs)

    # Create executor with retry policy
    retry_policy = RetryPolicy(
        max_retries=3,
        base_delay_seconds=5.0,
        max_delay_seconds=60.0,
    )

    executor = DistributedExecutor(
        workers=workers,
        storage=storage,
        retry_policy=retry_policy,
        verify_workers=True,
        poll_interval_seconds=2.0,
    )

    # Generate job ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"phase1a_{timestamp}"

    # Execute
    logger.info(f"  Starting job: {job_id}")
    job_result = executor.execute(
        task=task,
        inputs=inputs,
        job_id=job_id,
        artifact_dir=storage.artifact_dir(job_id),
    )

    # Convert TaskResults to the expected dict format
    results = []
    for tr in job_result.results:
        results.append({
            "index": tr.metrics.get("original_index", tr.index),
            "loss": tr.metrics.get("loss", 0.0),
            "loss_q": tr.metrics.get("loss_q", 0.0),
            "loss_a": tr.metrics.get("loss_a", 0.0),
            "converged": tr.metrics.get("converged", False),
            "num_restarts_tried": tr.metrics.get("num_restarts_tried", 0),
            "steps_to_converge": tr.metrics.get("steps_to_converge", 0),
            "content_path": str(tr.artifact_paths.get("content", "")) if tr.artifact_paths else "",
        })

    # Sort by original index
    results.sort(key=lambda x: x["index"])

    # Compute statistics
    converged_count = sum(1 for r in results if r["converged"])
    losses = [r["loss"] for r in results]
    losses_q = [r["loss_q"] for r in results]
    losses_a = [r["loss_a"] for r in results]

    random_baseline = baselines["random_combined_mean"]
    loss_ratios = [l / random_baseline for l in losses] if random_baseline > 0 else losses

    stats = {
        "total": len(results),
        "converged": converged_count,
        "convergence_rate": converged_count / len(results) if results else 0,
        "mean_loss": sum(losses) / len(losses) if losses else 0,
        "mean_loss_q": sum(losses_q) / len(losses_q) if losses_q else 0,
        "mean_loss_a": sum(losses_a) / len(losses_a) if losses_a else 0,
        "min_loss": min(losses) if losses else 0,
        "max_loss": max(losses) if losses else 0,
        "mean_loss_ratio_vs_random": sum(loss_ratios) / len(loss_ratios) if loss_ratios else 0,
        "mean_restarts": sum(r["num_restarts_tried"] for r in results) / len(results) if results else 0,
        "job_id": job_id,
        "elapsed_seconds": job_result.elapsed_time_seconds,
        "success_rate": job_result.success_rate,
        "failed_tasks": job_result.failed_tasks,
    }

    logger.info(f"\nStage 2 Results (experiment-runner - {len(workers)} workers):")
    logger.info(f"  Job ID: {job_id}")
    logger.info(f"  Elapsed: {job_result.elapsed_time_seconds:.1f}s")
    logger.info(f"  Convergence: {stats['convergence_rate']:.1%} ({converged_count}/{len(results)})")
    logger.info(f"  Mean loss: {stats['mean_loss']:.4f} (Q: {stats['mean_loss_q']:.4f}, A: {stats['mean_loss_a']:.4f})")
    logger.info(f"  Loss ratio vs random: {stats['mean_loss_ratio_vs_random']:.2f}x")
    logger.info(f"  Mean restarts needed: {stats['mean_restarts']:.1f}")
    if job_result.failed_tasks:
        logger.warning(f"  Failed tasks: {len(job_result.failed_tasks)} indices")

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

    # Experiment tracking setup (enabled by default if available)
    tracker_db = None
    run_id = None
    git_commit = get_git_commit()
    machine = get_hostname()

    if not args.no_track and TRACKING_AVAILABLE:
        tracker_db = get_tracker(warn_if_missing=True)
        if tracker_db:
            run_id = generate_run_id(args.project, args.experiment, machine)
            tracker_db.register_run(
                run_id,
                project=args.project,
                experiment=args.experiment,
                machine=machine,
                model_name=args.model,
                git_commit=git_commit,
                tags=args.tags,
                notes=args.notes,
            )
            logger.info(f"Tracking run: {run_id}")

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

    # Load tokenizer (always needed)
    logger.info(f"Loading tokenizer for: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    logger.info("  Tokenizer loaded.")

    # In parallel/distributed mode, skip loading full model in main process (workers load their own)
    # Just get the config to determine hidden size
    model = None
    if (args.parallel or args.distributed) and 2 in stages:
        mode_str = "Distributed" if args.distributed else "Parallel"
        logger.info(f"  {mode_str} mode: skipping model load in main process (workers will load)")
        from transformers import AutoConfig
        config_hf = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        d = config_hf.hidden_size
        logger.info(f"  Model hidden size (from config): {d}")
    else:
        # Load full model for non-parallel execution
        logger.info(f"Loading model: {args.model}")
        logger.info("  (this may take a few minutes on first run - downloading weights)")
        try:
            logger.info("  Loading model weights...")
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

    # Create mode embeddings using "Q" and "A" tokens (skip in parallel mode - workers create their own)
    Q_mode, A_mode = None, None
    if model is not None:
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
    tokenized_pairs = tokenize_pairs(pairs, tokenizer, device, add_eos=args.add_eos)
    if args.add_eos:
        logger.info(f"Added EOS token ({tokenizer.eos_token_id}) to all targets (teaches model to stop)")

    # Create optimization config
    config = OptimizationConfig(
        K=args.K,
        d=d,
        num_restarts=args.num_restarts,
        num_steps=args.num_steps,
        lr=args.lr,
        # LR schedule
        use_warm_restarts=args.lr_warm_restarts,
        warm_restart_t0=args.lr_restart_t0,
        warm_restart_t_mult=args.lr_restart_mult,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
        # EOS-based convergence
        use_eos_convergence=args.eos_convergence,
        eos_prob_threshold=args.eos_threshold,
        # Scheduled sampling (Priority 1)
        use_scheduled_sampling=args.scheduled_sampling,
        ss_initial_epsilon=args.ss_epsilon_start,
        ss_final_epsilon=args.ss_epsilon_end,
        ss_decay_type=args.ss_decay,
        ss_decay_k=args.ss_decay_k,
        ss_nudge_min_epsilon=args.ss_min_epsilon,
        # Bidirectional epsilon adjustment
        ss_nudge_up_on_degenerate=args.ss_nudge_up,
        ss_nudge_up_factor=args.ss_nudge_up_factor,
        ss_nudge_up_max_epsilon=args.ss_nudge_up_max,
        # Semantic loss
        use_semantic_loss=args.semantic_loss,
        semantic_loss_weight=args.semantic_weight,
        semantic_loss_temperature=args.semantic_temperature,
        semantic_sentence_weight=args.semantic_sentence_weight,
        eos_penalty_weight=args.eos_penalty_weight,
        # Unlikelihood (Priority 2)
        use_unlikelihood=args.unlikelihood,
        unlikelihood_alpha=args.unlikelihood_alpha,
    )
    logger.info(f"Optimization config: K={config.K}, steps={config.num_steps}, restarts={config.num_restarts}")
    if args.lr_warm_restarts:
        logger.info(f"  LR warm restarts: T_0={args.lr_restart_t0}, T_mult={args.lr_restart_mult}")
    if args.repetition_penalty != 1.0 or args.no_repeat_ngram > 0:
        logger.info(f"  Inference anti-repetition: penalty={args.repetition_penalty}, no_repeat_ngram={args.no_repeat_ngram}")
    if args.scheduled_sampling:
        logger.info(f"  Scheduled sampling: ε {args.ss_epsilon_start}→{args.ss_epsilon_end} ({args.ss_decay}), floor={args.ss_min_epsilon}")
        if args.ss_nudge_up:
            logger.info(f"  Bidirectional nudge: UP factor={args.ss_nudge_up_factor}, max={args.ss_nudge_up_max}")
    if args.semantic_loss:
        logger.info(f"  Semantic loss: weight={args.semantic_weight}, temp={args.semantic_temperature}, sentence={args.semantic_sentence_weight}, eos_penalty={args.eos_penalty_weight}")
    if args.unlikelihood:
        logger.info(f"  Unlikelihood training: α={args.unlikelihood_alpha}")

    # Compute baselines (skip in parallel mode - uses placeholder values)
    if model is not None:
        baselines = compute_baselines(
            model, tokenized_pairs, args.K, d, device,
            use_semantic_loss=args.semantic_loss,
            semantic_sentence_weight=args.semantic_sentence_weight,
            eos_token_id=tokenizer.eos_token_id,
            eos_penalty_weight=config.eos_penalty_weight,
        )
    else:
        # Parallel/distributed mode: use placeholder baselines (loss ratio won't be meaningful)
        logger.info("  Parallel/distributed mode: using placeholder baselines")
        baselines = {
            "random_q_mean": 1.0,
            "random_a_mean": 1.0,
            "random_combined_mean": 1.0,
            "loss_type": "semantic" if args.semantic_loss else "ce",
        }

    # Update loss threshold based on baselines
    # For CE loss: threshold = min(0.1, 0.01 * random_baseline)
    # For semantic loss: range is [0, 2], so use tighter threshold
    if args.semantic_loss:
        # Semantic loss: random embeddings should give ~0.5-1.0, good results < 0.1
        config.loss_threshold = min(0.1, baselines["random_combined_mean"] * 0.1)
    else:
        # CE loss: random embeddings give ~10-15, good results < 0.1
        config.loss_threshold = min(0.1, baselines["random_combined_mean"] * 0.01)
    logger.info(f"Dynamic loss threshold: {config.loss_threshold:.4f}")
    if args.eos_convergence:
        logger.info(f"EOS convergence: enabled (threshold P(EOS) >= {args.eos_threshold})")

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
    stage2_parallel_mode = False
    if 2 in stages:
        if args.parallel or args.distributed:
            stage2_parallel_mode = True
            # Convert config to dict for parallel/distributed workers
            config_dict = {
                "K": config.K,
                "d": config.d,
                "num_restarts": config.num_restarts,
                "num_steps": config.num_steps,
                "lr": config.lr,
                "use_warm_restarts": config.use_warm_restarts,
                "warm_restart_t0": config.warm_restart_t0,
                "warm_restart_t_mult": config.warm_restart_t_mult,
                "repetition_penalty": config.repetition_penalty,
                "no_repeat_ngram_size": config.no_repeat_ngram_size,
                "use_eos_convergence": config.use_eos_convergence,
                "eos_prob_threshold": config.eos_prob_threshold,
                "use_semantic_loss": config.use_semantic_loss,
                "semantic_loss_weight": config.semantic_loss_weight,
                "semantic_loss_temperature": config.semantic_loss_temperature,
                "semantic_sentence_weight": config.semantic_sentence_weight,
                "eos_penalty_weight": config.eos_penalty_weight,
                "use_scheduled_sampling": config.use_scheduled_sampling,
                "use_unlikelihood": config.use_unlikelihood,
                "unlikelihood_alpha": config.unlikelihood_alpha,
            }
            if args.distributed:
                if args.use_experiment_runner:
                    # Use new experiment-runner library with proper artifact handling
                    stage2_results, stage2_stats = run_stage2_experiment_runner(
                        args.distributed, args.model, pairs, config_dict,
                        args.add_eos, baselines, args.shared_fs,
                        args.remote_dir, args.remote_venv,
                    )
                else:
                    # Legacy distributed mode (JSON protocol, no artifacts)
                    stage2_results, stage2_stats = run_stage2_distributed(
                        args.distributed, args.model, pairs, config_dict,
                        args.add_eos, baselines, args.remote_dir, args.remote_venv,
                    )
            else:
                stage2_results, stage2_stats = run_stage2_parallel(
                    args.model, pairs, config_dict, args.add_eos, baselines
                )
        else:
            stage2_results, stage2_stats = run_stage2(
                model, tokenizer, tokenized_pairs, pairs, Q_mode, A_mode, config, device, baselines
            )

    # Stage 3: Spot check (requires stage 2 results)
    spot_checks = []
    if 3 in stages:
        if stage2_parallel_mode:
            logger.warning("Stage 3 not supported in parallel/distributed mode (no content tensors). Skipping.")
        elif stage2_results is None:
            logger.warning("Stage 3 requires Stage 2 results. Running Stage 2 first...")
            stage2_results, stage2_stats = run_stage2(
                model, tokenizer, tokenized_pairs, pairs, Q_mode, A_mode, config, device, baselines
            )
            spot_checks = run_stage3(
                model, tokenizer, stage2_results, pairs, Q_mode, A_mode, device
            )
        else:
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
            "use_warm_restarts": config.use_warm_restarts,
            "warm_restart_t0": config.warm_restart_t0,
            "warm_restart_t_mult": config.warm_restart_t_mult,
            "d": d,
            "add_eos": args.add_eos,
            "loss_threshold": config.loss_threshold,
            "repetition_penalty": config.repetition_penalty,
            "no_repeat_ngram_size": config.no_repeat_ngram_size,
            "use_scheduled_sampling": config.use_scheduled_sampling,
            "ss_initial_epsilon": config.ss_initial_epsilon,
            "ss_final_epsilon": config.ss_final_epsilon,
            "ss_decay_type": config.ss_decay_type,
            "ss_nudge_min_epsilon": config.ss_nudge_min_epsilon,
            "ss_nudge_up_on_degenerate": config.ss_nudge_up_on_degenerate,
            "ss_nudge_up_factor": config.ss_nudge_up_factor,
            "ss_nudge_up_max_epsilon": config.ss_nudge_up_max_epsilon,
            "use_semantic_loss": config.use_semantic_loss,
            "semantic_loss_weight": config.semantic_loss_weight,
            "semantic_loss_temperature": config.semantic_loss_temperature,
            "semantic_sentence_weight": config.semantic_sentence_weight,
            "eos_penalty_weight": config.eos_penalty_weight,
            "use_unlikelihood": config.use_unlikelihood,
            "unlikelihood_alpha": config.unlikelihood_alpha,
            "use_eos_convergence": config.use_eos_convergence,
            "eos_prob_threshold": config.eos_prob_threshold,
            "stages_run": sorted(stages),
            # Environment info for tracking
            "machine": machine,
            "git_commit": git_commit,
            "python_version": platform.python_version(),
        },
        "baselines": baselines,
        "stage1": stage1_detailed if stage1_detailed else stage1_summary,
        "stage2": stage2_stats,
        "stage2_per_example": (
            # Parallel mode returns dicts, normal mode returns OptimizationResult objects
            stage2_results if stage2_parallel_mode else [
                {
                    "loss": r.loss,
                    "loss_q": r.loss_q,
                    "loss_a": r.loss_a,
                    "converged": r.converged,
                    "num_restarts_tried": r.num_restarts_tried,
                    "steps_to_converge": r.steps_to_converge,
                }
                for r in stage2_results
            ]
        ) if stage2_results else None,
        "spot_checks": spot_checks if spot_checks else None,
        "timestamp": timestamp,
    }

    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")

    # Also save the optimized embeddings for successful cases (not available in parallel mode)
    if stage2_results and not stage2_parallel_mode:
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

    # Complete experiment tracking
    if tracker_db and run_id:
        try:
            convergence_rate = stage2_stats.get("convergence_rate") if stage2_stats else None
            mean_loss = stage2_stats.get("mean_loss") if stage2_stats else None
            tracker_db.complete_run(
                run_id,
                results_path=results_file,
                convergence_rate=convergence_rate,
                mean_loss=mean_loss,
                num_examples=args.num_examples,
            )
            logger.info(f"Tracking completed: {run_id}")
        except Exception as e:
            logger.warning(f"Failed to complete tracking: {e}")
        finally:
            tracker_db.close()


if __name__ == "__main__":
    main()
