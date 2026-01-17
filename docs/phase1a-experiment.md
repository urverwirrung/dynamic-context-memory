# Phase 1a Experimental Design Plan

## Hardware & Scope

- **Hardware**: 1x RTX 5090 + 2x RTX 5080 + 1x RTX 5070 Ti (cloud if necessary)
- **Scope**: Core validation only — minimal infrastructure to answer: "does joint Q-A optimization work?"

## Current State

- **Design docs**: Comprehensive (`architecture.md`, `phase1a-background.md`)
- **Code**: Core optimization loop exists (`phase1a_optimization.py`) but needs model integration
- **Missing**: Dataset loading, generator loss function implementation, experiment script

## Key Design Decisions

### 1. Generator Model Selection

The OSS landscape has expanded dramatically. Given your hardware (5090 ~32GB), here are the best current options:

| Model | Total Params | Active Params | VRAM | Notes |
|-------|--------------|---------------|------|-------|
| **Kimi K2** | 1T | 32B (MoE) | ~64GB | #1 open-source, beats GPT-5 on some benchmarks |
| **Kimi K2 (quantized)** | 1T | 32B (MoE) | ~24-32GB | Q4/Q5 quant may fit on 5090 |
| **DeepSeek-R1-Distill-14B** | 14B | 14B | ~28GB | R1 reasoning distilled into Qwen2.5, fits on 5090 |
| **DeepSeek-R1-Distill-7B** | 7B | 7B | ~14GB | R1 reasoning, very practical size |
| **Qwen2.5-7B** | 7B | 7B | ~14GB | Strong benchmarks, 128K context |
| **Qwen2.5-14B** | 14B | 14B | ~28GB | Better representations, still fits |
| **gpt-oss-20b** | 21B | 3.6B (MoE) | ~16GB | OpenAI open-weights, Apache 2.0 |
| **gpt-oss-120b** | 117B | 5.1B (MoE) | ~32GB | Near o4-mini quality, tight fit on 5090 |

**Key insight**: DeepSeek-R1-Distill models are particularly interesting — they have reasoning capabilities distilled from R1 (o1-competitive) into manageable dense models. The 7B/14B versions are practical for iteration.

**MoE consideration**: MoE models have many total parameters (rich representations) but only activate a subset per token. This could be advantageous—more "knowledge surface area" to address with soft prompts. Kimi K2 and gpt-oss are both MoE.

**Recommendation**:
1. **Primary**: DeepSeek-R1-Distill-7B — reasoning capabilities, fast iteration, well-tested
2. **Scale up**: DeepSeek-R1-Distill-14B or Qwen2.5-14B for better representations
3. **Stretch**: Kimi K2 (quantized) or gpt-oss-120b to test if MoE architecture helps

Sources: [Kimi K2 announcement](https://moonshotai.github.io/Kimi-K2/), [VentureBeat on Kimi K2](https://venturebeat.com/ai/moonshots-kimi-k2-thinking-emerges-as-leading-open-source-ai-outperforming), [DeepSeek-R1 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1), [DeepSeek models guide](https://www.bentoml.com/blog/the-complete-guide-to-deepseek-models-from-v3-to-r1-and-beyond)

### 2. Dataset Selection

**Tier 1 (Start here):**
- **SQuAD 2.0**: Clean extractive Q-A, answer is substring of context
  - Advantage: Simple, well-defined, answer derivable from question context
  - Risk: May be "too easy" - answer often lexically present

**Tier 2 (After Tier 1 success):**
- **TriviaQA**: Requires world knowledge beyond question text
  - Tests whether K encodes factual knowledge vs. just compression
- **Natural Questions**: Google search queries with Wikipedia answers
  - More realistic Q-A structure

**Tier 3 (Structural Reasoning):**
- **Sudoku**: Input = puzzle grid, Output = solved grid
  - Tests factorization beyond linguistic Q-A
  - Q and A are both structured (9x9 grids with constraints)
  - Requires the model to "know" Sudoku solving, not just pattern match
  - Easier than ARC-AGI: well-defined rules, single solution

- **Simple arithmetic**: "2+3=?" -> "5"
  - Minimal surface form, tests pure factorization
  - Baseline for procedural knowledge encoding

**Tier 4 (ARC-AGI — Stretch):**
- **ARC-AGI-1/2**: Abstract pattern completion from examples
  - Format: JSON with train demos (3 pairs) + test input
  - Pure LLMs score 0% on ARC-AGI-2 — very hard
  - *However*: Winning approaches use test-time training/adaptation
  - DCM's per-example optimization is philosophically aligned with what works
  - Each ARC task is like optimizing c* for that specific pattern

**Where structural reasoning fits in the experiment flow:**
```
Tier 1 (SQuAD)          ->  Validates basic joint optimization works
    |
Tier 2 (TriviaQA)       ->  Validates knowledge encoding
    |
Tier 3a (Arithmetic)    ->  Simple procedural knowledge
    |
Tier 3b (Sudoku)        ->  Structured I/O, constraint satisfaction
    |
Tier 4 (ARC-AGI)        ->  Abstract reasoning (if earlier tiers succeed)
```

**Recommendation**: Start with SQuAD (100 pairs), then arithmetic (sanity check), then Sudoku subset.

Sources: [ARC Prize 2025](https://arcprize.org/blog/arc-prize-2025-results-analysis), [ARC-AGI GitHub](https://github.com/fchollet/ARC-AGI)

### 3. K (Content Slots) Selection Strategy

The design doc suggests: "start with K large enough that optimization succeeds reliably, then reduce K until performance degrades."

**Proposed sweep:**
- K in {4, 8, 16, 32, 64, 128}
- For each K, measure: convergence rate, final loss, time to converge

**Information-theoretic intuition:**
- Typical Q: 10-20 tokens -> ~50-100 bits of info
- Typical A: 1-10 tokens -> ~5-50 bits
- DeepSeek-R1-Distill-7B (Qwen2.5 base): d=3584, so K=8 gives 8x3584 = 28,672 floats
- Even with low effective dimensionality, this should be sufficient for most Q-A pairs

**Recommendation**: Start with K=32 (likely overkill), establish baseline, then ablate down.

### 4. Staged Experimental Approach

```
Stage 0: Infrastructure
    |
Stage 1: Single-target diagnostic (Q only, A only)
    |
Stage 2: Joint Q-A optimization (core experiment)
    |
Stage 3: Diagnostic analysis
    |
Stage 4: Ablations and scaling
```

### 5. Success Criteria (Refined)

**Establishing baselines first:**
Before optimization, measure baseline losses to calibrate thresholds:
1. **Random embeddings baseline**: L(target | random_embeds) — this is the "no information" baseline
2. **Full context baseline**: L(A | Q) with Q as actual token prompt — this is the "best case" for the model
3. **Single token baseline**: L(target | single_random_token) — minimal prompt

**Per-pair convergence definition:**
```
converged = (final_loss < random_baseline * 0.3)  # 70% reduction from random
         OR (final_loss < full_context_baseline * 2.0)  # Within 2x of full context
```

The first criterion catches cases where we've found *some* useful structure.
The second criterion catches cases where we're approaching the model's actual capability.

**Loss curve health indicators:**
- OK Healthy: Monotonic decrease with diminishing returns
- OK Healthy: Initial plateau then sudden drop (finding the basin)
- WARN Warning: Oscillation (lr too high, or conflicting Q/A gradients)
- FAIL Failure: Flat from start (model not responding to soft prompts)
- FAIL Failure: Decreasing then diverging (unstable optimization)

**Dataset-level thresholds:**
| Metric | Minimum Viable | Good | Excellent |
|--------|----------------|------|-----------|
| Convergence rate | >40% | >70% | >90% |
| Mean loss ratio (vs random) | <0.4 | <0.25 | <0.15 |
| Q/A loss balance | both < 2x baseline | both < 1.5x | both < 1.2x |

**Qualitative validation (Stage 3):**
For converged examples, greedy decode from (mode, c*):
- **Pass**: Output is semantically equivalent to target (may differ in phrasing)
- **Partial**: Output captures key information but has errors/omissions
- **Fail**: Output is unrelated or gibberish

Target: >80% Pass rate among converged examples. If convergence rate is high but Pass rate is low, the optimization is finding degenerate solutions.

**Early stopping signals:**
- If <20% convergence after 50 pairs -> stop, investigate model/hyperparameters
- If Q loss and A loss anti-correlate -> deploy semantic equivalence regularization
- If loss is 0.0 for many examples -> check for data leakage or degenerate solutions

### 6. Semantic Equivalence Regularization (Fallback Strategy)

If joint Q-A optimization struggles with gradient interference, use a strong model to generate paraphrases:

**The idea** (from architecture.md):
```
{Q1, Q2, ..., Qn} = paraphrases of Q
{A1, A2, ..., Am} = paraphrases of A

Objective: c* = argmin_c [ sum_i L(Qi | Q_mode, c) + sum_j L(Aj | A_mode, c) ]
```

**Why this helps**:
- Gradients from surface-level idiosyncrasies (word choice, phrasing) cancel across paraphrases
- Gradients from shared semantic structure reinforce
- Reduces interference between Q and A optimization targets
- Analogous to data augmentation in vision

**Implementation**:
1. Use Claude/GPT-4 to generate 3-5 paraphrases of each Q and A
2. Store as extended dataset: `(Q_variants, A_variants)` per example
3. Modify loss computation to sum over all variants
4. May need to weight original vs. paraphrases

**When to deploy**:
- If joint optimization converges <30% of the time
- If Q and A losses fight each other (one goes down, other goes up)
- As a diagnostic: if paraphrases help dramatically, surface form was the problem

---

## Experimental Flow (Core Validation)

### Stage 1: Single-Target Diagnostic

**Purpose**: Validate the Generator's embedding space is addressable before attempting dual-mode.

```python
# Can we reconstruct Q alone? A alone?
for target in [sample_questions, sample_answers]:
    result = SingleTargetOptimizer.run_diagnostic(target)
```

**Decision gate**: If <70% convergence on single targets, stop and investigate.

### Stage 2: Joint Q-A Optimization

**Core experiment**: 100 SQuAD pairs with K=32, 5 restarts, 500 steps.

**Metrics:**
- Convergence rate (target: >50%)
- Final loss (Q and A separately)
- Restarts needed

**Decision gate**: If <30% convergence, the approach may not work with this model/dataset.

### Stage 3: Spot Check (if Stage 2 passes)

For 5 successful cases:
1. Greedy decode from (Q_mode, c*) — is output recognizable as Q?
2. Greedy decode from (A_mode, c*) — is output recognizable as A?

This validates that low loss actually means good generation, not degenerate optimization.

---

## Follow-on Experiments (if core validation succeeds)

Deferred until after core validation:
- K ablation (find minimum viable K)
- Dataset expansion (TriviaQA, HotpotQA)
- Model comparison (8B vs 3B)
- Embedding space analysis (clustering, distance from token manifold)
- Loss landscape characterization

---

## Minimal Implementation (Core Validation Scope)

Given "core validation only" scope, we need just enough to answer the key question.

### Files to Create/Modify

```
dynamic-context-memory/
|-- phase1a_optimization.py      # Existing - add generator_loss_fn for HF models
|-- run_phase1a.py               # NEW: Single script for the experiment
+-- requirements.txt             # NEW: Dependencies
```

That's it. One new script, one function addition, one requirements file.

### `run_phase1a.py` Structure

```python
# 1. Load model (DeepSeek-R1-Distill-7B or configurable)
# 2. Load small dataset (100 SQuAD pairs)
# 3. Run single-target diagnostic (Stage 1)
# 4. If Stage 1 passes: Run joint Q-A optimization (Stage 2)
# 5. Print summary statistics
# 6. Save results to JSON for later analysis
```

### Key Implementation Detail: Soft Prompt Injection

The `generator_loss_fn` needs to:
1. Take optimizable embeddings `c` of shape (K, d)
2. Prepend mode embedding: `[mode, c]` -> shape (K+1, d)
3. Embed target tokens: `embed(target)` -> shape (seq_len, d)
4. Forward through model with `inputs_embeds=concat(prompt, target_embeds)`
5. Compute cross-entropy loss on target positions only

**Concrete HuggingFace implementation:**

```python
def generator_loss_fn(
    model: AutoModelForCausalLM,
    target_tokens: torch.Tensor,      # (seq_len,) - the Q or A tokens
    prompt_embeds: torch.Tensor,      # (K+1, d) - mode + content embeddings
) -> torch.Tensor:
    """
    Compute cross-entropy loss for generating target_tokens given prompt_embeds.

    The model sees: [prompt_embeds | target_embeds]
    We compute loss only on the target positions (shifted by 1 for autoregressive).
    """
    # Get the embedding layer
    embed_layer = model.get_input_embeddings()

    # Embed target tokens
    target_embeds = embed_layer(target_tokens)  # (seq_len, d)

    # Concatenate: prompt + target
    # Shape: (1, K+1+seq_len, d)
    full_embeds = torch.cat([prompt_embeds, target_embeds], dim=0).unsqueeze(0)

    # Create labels: -100 for prompt positions (ignored in loss), actual tokens for target
    # For autoregressive LM: predict token[i+1] from position[i]
    # So labels are shifted: we want to predict target_tokens from the last prompt position onward
    prompt_len = prompt_embeds.shape[0]

    # Labels: [-100, -100, ..., -100, target[0], target[1], ..., target[n-1]]
    # But for causal LM, labels[i] = what to predict at position i
    # Position prompt_len-1 should predict target[0]
    # Position prompt_len should predict target[1], etc.

    labels = torch.full((1, full_embeds.shape[1]), -100, dtype=torch.long, device=target_tokens.device)
    labels[0, prompt_len:] = target_tokens  # Predict target tokens starting after prompt

    # Forward pass
    outputs = model(inputs_embeds=full_embeds, labels=labels)

    return outputs.loss
```

### Dependencies

```
torch>=2.0
transformers>=4.40
datasets
tqdm
```

---

## Verification Plan

1. **Sanity check**: Confirm loss decreases during optimization (not flat/increasing)
2. **Single-target baseline**: Confirm we can reconstruct Q alone, A alone
3. **Joint optimization**: Measure convergence rate on 100 SQuAD pairs
4. **Spot check**: Manually inspect 5 successful cases — does greedy decode produce recognizable Q/A?

**Success threshold**: >50% convergence rate on joint Q-A with K=32 -> proceed to deeper analysis

---

*Document created: 2026-01-17*
