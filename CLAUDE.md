# Dynamic Context Memory - Project Context

## What This Is

Research project exploring a novel transformer architecture where:
- A fixed-size embedding buffer (STM) replaces growing KV cache
- A "Diffuser" network learns to iteratively refine STM via diffusion-like process
- A frozen pretrained LLM (Generator) produces outputs from STM soft prompts

**Core hypothesis**: Smaller, dynamically managed context yields higher quality at lower cost than traditional attention over full context.

## Current Phase: 1a - Target Generation via Optimization

Validating whether we can optimize STM content such that a single content representation works for both Q and A outputs:

```
[Q_mode | content] → Generator → Q tokens
[A_mode | content] → Generator → A tokens
```

**Stages**:
1. Single-target diagnostic (Q alone, A alone) - validates addressability
2. Joint Q-A optimization - core validation
3. Spot validation - greedy decode verification

## Key Files

| File | Purpose |
|------|---------|
| `run_phase1a.py` | Main orchestrator - CLI args, model loading, stage execution |
| `phase1a_optimization.py` | Core optimization loop - ContentOptimizer, loss computation |
| `phase1a_worker.py` | Distributed worker - JSON protocol for multi-GPU/machine |
| `analyze_results.py` | Results analysis and visualization |
| `docs/architecture.md` | Full v0.6 design document |
| `docs/phase1a-experiment.md` | Experimental design and staging |

## Running Experiments

```bash
# Basic run (Stage 2 only)
python run_phase1a.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                      --num-examples 100 --K 32 --num-steps 800

# All stages
python run_phase1a.py --all

# Distributed across GPUs (device:weight format)
python run_phase1a.py --distributed "localhost:0:100,localhost:1:70" --num-examples 200
```

## Architecture (v1)

```
Input tokens → Input Encoder → embeddings
                                  ↓
                         [Cross-attention]
                                  ↓
Noise → Diffuser ←────── STM ──→ Generator → Output tokens
```

**STM**: (K+1, d) tensor - K content embeddings + 1 mode embedding (Q or A)

## Known Issues & Solutions

| Issue | Description | Mitigation |
|-------|-------------|------------|
| Repetition | Correct output then continues ("ATSC→ATSCATSCATSC") | EOS convergence criterion, length constraints |
| Semantic equivalents | Paraphrase variants don't match exactly | Semantic loss (cosine similarity blend) |
| Teacher forcing gap | Training/inference behavior mismatch | Scheduled sampling |

## Key Hyperparameters

- `K`: Content slots (default: 32, range: 4-128)
- `num_steps`: Optimization steps per restart (default: 800)
- `num_restarts`: Maximum restarts before giving up (default: 5)
- `lr`: Learning rate (default: 1.0, relatively high for soft prompt optimization)

## Dataset

Primary: SQuAD 2.0 (100 pairs for validation)
Future: TriviaQA, arithmetic tasks, ARC-AGI

## Success Criteria

- Convergence rate: >50% (loss < 0.3x random baseline OR < 2.0x full context)
- Both Q and A loss within 2x baseline
- Qualitative: >80% converged examples produce correct outputs

## Development Notes

- Results stored in `results/` as JSON + PyTorch tensors
- Real-time logging to stderr during optimization
- Supports multi-GPU via accelerate and custom distributed protocol
