#!/usr/bin/env python3
"""
Standalone worker for distributed Phase 1a Stage 2 optimization.

Receives work via stdin (JSON), processes it, outputs results to stdout (JSON).
Can be run locally or via SSH on remote machines.

Usage:
    echo '{"gpu_id": 0, "model_name": "...", "pairs": [...], ...}' | python phase1a_worker.py
"""

import json
import sys
import torch
from typing import List, Tuple, Dict

def run_worker(config: Dict) -> List[Dict]:
    """Run optimization on assigned examples."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from phase1a_optimization import (
        ContentOptimizer, OptimizationConfig, create_mode_embeddings,
        hf_generator_loss_fn,
    )
    from tqdm import tqdm

    gpu_id = config["gpu_id"]
    model_name = config["model_name"]
    pairs = config["pairs"]  # List of [q, a] pairs
    indices = config["indices"]
    config_dict = config["config"]
    add_eos = config["add_eos"]
    worker_id = config.get("worker_id", 0)

    device = torch.device(f"cuda:{gpu_id}")

    # Load model
    print(f"[Worker {worker_id} / GPU {gpu_id}] Loading model...", file=sys.stderr)
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

    # Create mode embeddings
    q_token_id = tokenizer.encode("Q", add_special_tokens=False)[0]
    a_token_id = tokenizer.encode("A", add_special_tokens=False)[0]
    Q_mode, A_mode = create_mode_embeddings(model, q_token_id, a_token_id)
    Q_mode = Q_mode.to(device)
    A_mode = A_mode.to(device)

    # Tokenize pairs
    tokenized = []
    eos_token_id = tokenizer.eos_token_id
    for q, a in pairs:
        q_tokens = tokenizer.encode(q, return_tensors="pt", add_special_tokens=False).squeeze(0).to(device)
        a_tokens = tokenizer.encode(a, return_tensors="pt", add_special_tokens=False).squeeze(0).to(device)
        if add_eos and eos_token_id is not None:
            eos_tensor = torch.tensor([eos_token_id], device=device)
            q_tokens = torch.cat([q_tokens, eos_tensor])
            a_tokens = torch.cat([a_tokens, eos_tensor])
        tokenized.append((q_tokens, a_tokens))

    # Create optimizer config
    opt_config = OptimizationConfig(**config_dict)

    # Create optimizer
    optimizer = ContentOptimizer(
        generator=model,
        generator_loss_fn=hf_generator_loss_fn,
        Q_mode=Q_mode,
        A_mode=A_mode,
        config=opt_config,
        device=device,
        eos_token_id=tokenizer.eos_token_id,
        tokenizer=tokenizer,
    )

    # Run optimization
    results = []
    num_examples = len(tokenized)
    total_steps = num_examples * opt_config.num_restarts * opt_config.num_steps

    print(f"[Worker {worker_id} / GPU {gpu_id}] Processing {num_examples} examples...", file=sys.stderr)
    with tqdm(total=total_steps, desc=f"W{worker_id}/GPU{gpu_id}", file=sys.stderr) as pbar:
        for i, (q_tokens, a_tokens) in enumerate(tokenized):
            result = optimizer.optimize(
                q_tokens, a_tokens,
                pbar=pbar, example_idx=i, num_examples=num_examples,
            )
            original_idx = indices[i]
            results.append({
                "index": original_idx,
                "loss": result.loss,
                "loss_q": result.loss_q,
                "loss_a": result.loss_a,
                "converged": result.converged,
                "num_restarts_tried": result.num_restarts_tried,
                "steps_to_converge": result.steps_to_converge,
            })

            status = "✓" if result.converged else "✗"
            print(f"  {status} [{original_idx}] W{worker_id}/GPU{gpu_id} loss={result.loss:.6f}", file=sys.stderr)

    print(f"[Worker {worker_id} / GPU {gpu_id}] Done. Converged: {sum(1 for r in results if r['converged'])}/{len(results)}", file=sys.stderr)
    return results


def main():
    # Read config from stdin
    input_data = sys.stdin.read()
    config = json.loads(input_data)

    # Run worker
    results = run_worker(config)

    # Output results to stdout as JSON
    print(json.dumps(results))


if __name__ == "__main__":
    main()
