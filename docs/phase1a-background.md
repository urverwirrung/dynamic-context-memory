# Phase 1a: Target Generation via Optimization

## Context: Dynamic Context Memory Architecture

This document focuses on **Phase 1a** of the Dynamic Context Memory (DCM) project. Phase 1a is the critical first step that validates whether the entire approach is viable before investing in later phases.

### The Core Insight

DCM treats context management as an **iterative refinement process** analogous to image diffusion. Instead of monotonically accumulating tokens in a KV cache, we maintain a fixed-size buffer of continuous embeddings (Short-Term Memory / STM) that gets refined to optimize downstream task performance.

The key mechanism: a **Diffuser** network iteratively refines STM embeddings, conditioned on input via cross-attention. The STM then serves as a soft prompt to a frozen Generator LLM.

### What Phase 1a Validates

Phase 1a answers a foundational question: **Can we generate viable training targets for the Diffuser?**

The Diffuser needs to learn a mapping from (noisy STM + input) → optimal STM. But what is "optimal STM"? We define it operationally:

> **Optimal STM embeddings are those that minimize the Generator's loss when generating both the question Q and answer A.**

This is the optimization problem at the heart of Phase 1a.

### The Optimization Problem

Given a (Q, A) pair, a frozen Generator, and fixed mode embeddings, find content embeddings that minimize generation loss for both Q and A:

```
content* = argmin_c [ L_gen(Q | concat(Q_mode, c)) + L_gen(A | concat(A_mode, c)) ]

where:
  c: tensor of shape (K, d) — the content embeddings to optimize
  Q_mode, A_mode: fixed embeddings for mode selection (arbitrary distinct tokens)
  L_gen: Generator's cross-entropy loss on target sequence
```

### Why Joint Q-A Optimization

Requiring the same content to support generation of *both* Q and A forces the representation to encode their shared structure. If optimization succeeds:
- The Generator has factored representations where Q and A share structure
- The K embeddings function as an "address" into the Generator's implicit knowledge
- Genuine compression is occurring

If optimization fails:
- K is too small (information bottleneck)
- Q and A lack shared structure the Generator can exploit
- The Generator doesn't represent this knowledge in a factored way

### The Mode Mechanism

The STM has shape (K+1, d): K content positions + 1 mode position.

- **Mode position**: Fixed embedding (e.g., token "Q" or "A") that routes Generator output
- **Content positions**: The embeddings we optimize

During optimization, only content positions are updated; mode stays fixed. During inference, mode selects which output (Q-style or A-style) to generate from the shared content.

### Phase 1a as Diagnostic

Phase 1a itself reveals whether DCM is viable before any Diffuser training:

| Observable | What it tells you |
|------------|-------------------|
| **Optimization success rate** | What fraction of pairs converge? Widespread failure suggests the Generator lacks factored representations. |
| **Landscape difficulty** | Do different initializations converge similarly? Forgiving landscapes suggest robust structure. |
| **Distance from M_pretrain** | How far from nearest token embeddings? Large drift means heavy reliance on Generator extrapolation. |
| **Clustering structure** | Do related pairs cluster? Meaningful clustering suggests genuine semantic structure. |
| **Solution heterogeneity** | Qualitatively similar or wildly different? High heterogeneity means complex Diffuser task. |

### Relationship to Later Phases

```
Phase 1a: Target Generation    →  Produces (Q, A, content*) triples
    ↓
Phase 1b: K Backprop Diagnostic →  Tests if bridging knowledge is implicitly encoded
    ↓
Phase 2: Diffusion Training     →  Trains Diffuser to produce content* from noise + input
    ↓
v2+: Full Architecture          →  Adds LTM, streaming, gating
```

If Phase 1a fails, later phases won't succeed. If Phase 1a succeeds, we have validated training targets and can proceed with confidence.

---

## Prior Art and Techniques

The optimization problem in Phase 1a has close analogues in existing work. This section covers relevant prior art and techniques to borrow.

## Textual Inversion as Prior Art

Textual Inversion (Gal et al., 2022) solves an analogous problem in image diffusion: given images of a concept, find an embedding `v*` such that the diffusion model generates that concept when conditioned on `v*`.

### Structural Parallel

| | Textual Inversion | Phase 1a |
|-|-------------------|----------|
| **Objective** | `v* = argmin_v E[‖ε - ε_θ(x_t, t, c(v))‖²]` | `c* = argmin_c [L(Q|Q_mode,c) + L(A|A_mode,c)]` |
| **Optimized** | Single embedding | K embeddings |
| **Frozen model** | Diffusion U-Net | Generator LLM |
| **Conditioning** | Cross-attention | Soft prompt |
| **Interpretation** | Address into visual concept space | Address into Generator's factored knowledge |

**Core shared insight**: The optimized embedding doesn't store the target — it stores *where to find it* in the model's implicit structure.

### Key Difference

Phase 1a is harder: (1) reconstructs exact sequences vs. a distribution, (2) requires dual-mode routing (Q and A share content), testing that the Generator's knowledge is factored.

### Techniques to Borrow

**Regularization toward token manifold**
```python
loss = reconstruction_loss + λ * dist_to_nearest_token(c)
```
Use if optimization finds degenerate/OOD solutions.

**Progressive optimization**
- Phase A: 200 steps, lr=0.05 (coarse)
- Phase B: 300 steps, lr=0.005 from Phase A (refine)

**Diagnostic: single-target first**
```python
c_q = optimize(L_gen(Q | c))  # Single slot, Q only
```
If this fails, the problem is fundamental. If it succeeds, the joint Q-A objective is the harder extension worth pursuing.

## Soft Prompt Tuning as Prior Art

Soft prompt tuning (Lester et al., 2021; Li & Liang, 2021) learns continuous embeddings prepended to inputs that steer a frozen LM toward a task.

### Structural Parallel

| | Soft Prompt Tuning | Phase 1a |
|-|-------------------|----------|
| **Objective** | `p* = argmin_p Σᵢ L(yᵢ | xᵢ, p)` | `c* = argmin_c [L(Q|c) + L(A|c)]` |
| **Scope** | One `p*` for entire task | Different `c*` per (Q, A) |
| **What it encodes** | "How to do this task" | "What this Q-A is about" |
| **Training signal** | Many examples | Single (Q, A) pair |

**Core shared insight**: M_useful >> M_pretrain. Frozen models respond coherently to continuous embeddings never seen during pretraining.

### Key Difference

Soft prompts **generalize** (same p for all examples). Phase 1a **reconstructs** (optimal c per example). Phase 1a should achieve lower per-example loss than any task-level prompt — if not, optimization is failing.

### Findings That Transfer

| Finding | Implication for Phase 1a |
|---------|-------------------------|
| Random init works | Default baseline viable |
| Token count scales with complexity | K should scale with Q-A complexity |
| Larger models = easier optimization | Expect harder optimization on smaller generators |
| Task vectors exist | Content vectors should exist analogously |

### Techniques to Borrow

**Reparameterization** (if direct optimization unstable)
```python
c = mlp(latent_code)  # Optimize latent_code, not c
```

**Diagnostic: compare to task-level prompt**
```python
p_task = soft_prompt_train(dataset)  # Standard prompt tuning
for Q, A in dataset:
    loss_task = L(A | Q, p_task)
    loss_instance = L(A | A_mode, c*)
    assert loss_instance < loss_task  # Must hold
```

### References

- Lester et al. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning"
- Li & Liang (2021). "Prefix-Tuning: Optimizing Continuous Prompts for Generation"
- Liu et al. (2023). "GPT Understands, Too" (P-Tuning)
- Gal et al. (2022). "An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion"
- Mokady et al. (2023). "Null-text Inversion for Editing Real Images using Guided Diffusion Models"
