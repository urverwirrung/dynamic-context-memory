    # Dynamic Context Memory Architecture
## Design Document v0.6

> **Scope**: This is a conceptual design document, not an implementation spec. Architecture choices are deliberately open where noted. The goal is to pin down component responsibilities and information flow precisely enough to guide implementation.

---

## Part I: Motivation and Core Insight

### Problem Statement

Current transformer architectures rely on static KV caching, leading to:
- Context bloat: monotonic accumulation without selectivity
- Attention degradation: signal dilution as context grows
- No agency: models cannot choose what to attend to
- Memory-bound inference: KV cache size limits concurrency

**Hypothesis**: A smaller, dynamically managed context with learned memory operations would yield higher quality responses at lower cost.

### The Diffusion Analogy

The core insight is that context management is an *iterative refinement process*, analogous to image diffusion:

| Image Diffusion | Context Diffusion |
|-----------------|-------------------|
| Pixel canvas | STM embedding buffer |
| Noise to remove | Contextual suboptimality (irrelevant, redundant, low-value content) |
| Conditioning signal | Input token embeddings (via cross-attention) |
| Target distribution | "Contexts that lead to good responses" |
| Denoised target | Optimized embeddings that minimize generation loss |
| Output | Refined embedding buffer |

#### The DALL-E / Stable Diffusion Parallel

Text-to-image diffusion provides a concrete architectural template:

| Text-to-Image | Input-to-STM |
|---------------|--------------|
| Text prompt | Input tokens (question, context) |
| CLIP/T5 encoder | Input Encoder (embeds input for conditioning) |
| Text embeddings | Input embeddings |
| U-Net with cross-attention | Diffuser with cross-attention |
| Pixel/latent canvas | STM embedding buffer |
| Generated image | Optimized context for Generator |

The key mechanism is **cross-attention**: at each denoising step, the STM embeddings (queries) attend to input embeddings (keys/values). Each position in the STM learns which parts of the input are relevant to it. This is exactly how text conditions image generation — we repurpose it for input conditioning context generation.

### Relationship to Prior Work

This architecture shares principles with Hopfield Retrieval Memory (HRM) and Transformer Retrieval Memory (TRM) approaches, particularly:
- **Deep supervision**: Training signal at intermediate steps, not just terminal output
- **Detached forward passes**: Carrying state forward as initialization while detaching from the computation graph between supervision steps

The key difference is in how state evolution is modeled:
- **TRM**: Learns a mapping function between states (state_t → state_{t+1})
- **Our approach**: Models the *incremental refinement process* via diffusion — predicting what change to apply at each step rather than the final state directly

This is analogous to how image diffusion models the denoising trajectory rather than learning a direct noise→image mapping. The diffusion framing provides a principled way to handle the iterative nature of context optimization.

---

## Part II: Architecture Overview

This section introduces all architectural components conceptually. Part III covers the v1 experimental scope (which uses a subset), and Part IV covers the full v2+ architecture.

### Component Summary

| Component | Purpose | v1 | v2+ |
|-----------|---------|:--:|:---:|
| **STM** | Fixed-size buffer of continuous embeddings; Generator input | ✓ | ✓ |
| **Input Encoder** | Embeds input tokens for Diffuser cross-attention | ✓ | ✓ |
| **Diffuser** | Iteratively refines STM conditioned on input | ✓ | ✓ |
| **Generator** | Produces output tokens from STM | ✓ | ✓ |
| **LTM** | Long-term storage of compressed STM states | | ✓ |
| **STM Encoder** | Embeds STM states for LTM retrieval/gating | | ✓ |
| **Gating Model** | Decides what to commit to LTM | | ✓ |

### Short-Term Memory (STM)

- **Representation**: Fixed-size buffer of continuous embeddings, shape (N, d)
- **Constraint**: Small capacity forces compression and selectivity
- **Interface**: Direct input to Generator (replaces standard token embeddings)
- **Key property**: Positions are "slots" holding continuous embeddings, not discrete tokens. No discretization occurs until the Generator produces output.

### Input Encoder (v1+)

- **Input**: Input tokens (question, context, etc.)
- **Output**: Sequence of embeddings for Diffuser cross-attention conditioning
- **v1 simplification**: Can be the Generator's own embedding layer
- **Purpose**: Provides the conditioning signal that guides STM refinement

### Diffuser (the novel component)

- **Input**: Current STM state, input embeddings (via cross-attention), noise level (during training)
- **Output**: Refined STM state
- **Mechanism**: Cross-attention over input embeddings at each layer
- **Scope**: Only transforms STM given conditioning. Does NOT do retrieval or storage decisions.

**Cross-attention mechanism**:
```
Q = project_q(stm_features)        # (N, d_attn)
K = project_k(input_embeddings)    # (input_len, d_attn)
V = project_v(input_embeddings)    # (input_len, d_attn)

attention = softmax(Q @ K.T / sqrt(d))  # (N, input_len)
output = attention @ V                   # (N, d_attn)
```

Each STM position learns to attend to relevant parts of the input.

### Generator

- **Input**: STM embeddings (as if they were token embeddings)
- **Output**: Response tokens (standard autoregressive generation)
- **v1**: Frozen pretrained LLM
- **Future**: Potentially fine-tuned on soft prompt inputs

### Long-Term Memory (LTM) — *v2+*

- **Storage**: Embeddings (not tokens)
- **Content**: Lossy compression of prior STM states
- **Structure**: Learned subspaces for (content, effect, timing)
- **Compression**: Variable temporal granularity via learned gating

### STM Encoder — *v2+*

- **Input**: Current STM state (the embedding buffer)
- **Output**: Single embedding vector
- **Purpose**: Dual-purpose — feeds both LTM retrieval (query) and Gating Model (commit decision input)

**Note**: This is distinct from the Input Encoder. The Input Encoder embeds *tokens* for Diffuser conditioning. The STM Encoder embeds *STM states* for LTM operations. These operate on different inputs and serve different purposes.

### Gating Model — *v2+*

- **Input**: STM embedding (from STM Encoder)
- **Output**: Write decision (discard / accumulate / commit buffer / commit solo)
- **Purpose**: Learned control over memory compression and persistence

---

## Part III: v1 — Core Hypothesis Validation

v1 isolates the core hypothesis: **can a diffusion process learn to produce task-optimal context representations?**

### v1 Scope

#### In Scope

- Simple Q-A datasets (SQuAD, TriviaQA, natural questions)
- Structural reasoning tasks (Sudoku, ARC-AGI) — tests whether compression generalizes beyond linguistic Q-A to constraint satisfaction and pattern recognition
- Complete input passed to Diffuser as conditioning signal (no streaming)
- Fixed STM size
- Frozen pretrained Generator
- Standard Gaussian diffusion in continuous embedding space
- Cross-attention conditioning (input → STM positions)

#### Deferred to v2+

- Long-Term Memory (LTM): retrieval, storage, gating
- Streaming/incremental input processing (token-by-token STM updates)
- Adaptive conditioning window
- Joint Generator training
- Variable STM size
- Multi-turn conversation handling

#### Why This Scoping

The Diffuser training signal is the critical path. By eliminating LTM and streaming, we can validate whether optimization-generated targets provide viable training signal before adding architectural complexity. If v1 fails, LTM won't save it. If v1 succeeds, LTM becomes a natural extension.

v1 uses complete input for cross-attention conditioning, sidestepping the question of how STM should evolve as tokens arrive incrementally. Streaming introduces additional complexity: how many diffusion steps per token? How does partial input affect target generation? These are important questions, but orthogonal to the core hypothesis. If optimization-based targets work with complete input, extending to streaming becomes a well-motivated v2 problem.

Moreover, v1 has a natural staged validation: Phase 1 (target generation) can reveal feasibility problems before any Diffuser training occurs.

### v1 Architecture

```
    Input tokens ──► Input Encoder ──► input_embeddings
                                            │
                                            ▼ cross-attention
                                      ┌──────────┐
    noise (training) / ──────────────►│ Diffuser │───► STM_refined
    pure noise (inference)            │          │
                                      └──────────┘
                                            │
                                            ▼
                                    ┌─────────────┐
                                    │  Generator  │───► Output tokens
                                    │  (frozen)   │
                                    └─────────────┘
```

#### STM Structure for v1

For v1, STM has shape (K+1, d) where:
- **K content positions**: Optimized by the Diffuser to encode task-relevant information
- **1 mode position**: Fixed embedding that routes Generator output (explained in Training Procedure below)

The content size K is a hyperparameter that controls the compression ratio. 

**Note on K selection**: The initial choice of K should be informed by:
- Information-theoretic lower bounds: K × d bits must encode at least I(Q) + I(A) - I(Q;A)
- Empirical pilot studies: vary K and measure optimization success rate
- Task complexity: structural tasks (Sudoku) may require different K than linguistic Q-A

A systematic approach: start with K large enough that optimization succeeds reliably, then reduce K until performance degrades. The minimum viable K reveals the true compression achievable for each task type.

### Target Generation via Optimization

This is the key insight enabling Diffuser training: **the "denoised" target for context diffusion is defined by optimization against downstream task performance.**

#### The Optimization Problem

Given a (Q, A) pair, a frozen Generator, and fixed mode embeddings, find content embeddings that minimize generation loss for both Q and A:

```
content* = argmin_c [ L_gen(Q | concat(Q_mode, c)) + L_gen(A | concat(A_mode, c)) ]

where:
  c: tensor of shape (K, d) — the content embeddings to optimize
  Q_mode, A_mode: fixed embeddings for mode selection (arbitrary distinct tokens)
  L_gen: Generator's cross-entropy loss on target sequence
```

#### Why Joint Q-A Optimization

Requiring the same content to support generation of *both* Q and A forces the representation to encode their shared structure. If optimization succeeds:
- The Generator has factored representations where Q and A share structure
- The K embeddings function as an "address" into the Generator's implicit knowledge
- Genuine compression is occurring

If optimization fails:
- K is too small (information bottleneck)
- Q and A lack shared structure the Generator can exploit
- The Generator doesn't represent this knowledge in a factored way

#### What Success Would Reveal

Successful optimization would demonstrate that the Generator has implicitly learned **factored representations** — separating semantic content from surface expression. A shared compressed state can route to different natural language realizations via mode selection alone.

This is stronger than observing that transformers have bidirectional information flow (tokens → latents → tokens). The novel finding would be that Q and A *share* a compressed representation, suggesting the model learned to separate "what the fact is" from "how to express it" — despite never being trained explicitly for this factorization.

#### What Optimal STMs Will Look Like

Optimal STM embeddings will probably not correspond to natural language. They'll be whatever continuous vectors best compress the Q-A relationship — likely appearing as "gibberish" if projected to nearest tokens.

This is desirable. We want the system to discover efficient encodings unconstrained by human-readable intermediate representations.

#### Regularization via Semantic Equivalence Classes

A potential regularization technique: have the Generator produce multiple paraphrases of Q and A, forming **semantic equivalence classes** {Q₁...Qₙ} and {A₁...Aₘ}.

Optimize over the equivalence class:
```
content* = argmin_c [ Σᵢ L_gen(Qᵢ | Q_mode, c) + Σⱼ L_gen(Aⱼ | A_mode, c) ]
```

This regularizes toward **paraphrase-invariant representations**:
- Gradients from surface-level idiosyncrasies (word choice, phrasing) cancel across paraphrases
- Gradients from shared semantic structure reinforce
- Result: higher probability of finding regions encoding **semantic invariants** rather than surface artifacts

This is analogous to data augmentation in vision — by requiring invariance across semantically equivalent inputs, we push representations toward capturing meaning rather than form.

#### Alternative: Oracle Distillation

An alternative approach is **distillation from a full-context oracle**. A large model with complete context generates ideal outputs; the Diffuser learns to produce STM that enables matching performance despite compression.

| Aspect | Optimization-Based | Oracle Distillation |
|--------|-------------------|---------------------|
| Target definition | Implicit (minimize loss) | Explicit (match oracle) |
| Compute per example | High (many optimization steps) | Lower (single forward pass) |
| Requires oracle model | No | Yes |
| Guarantees | Targets provably minimize Generator loss | Targets match oracle; hope this transfers |

**Recommendation**: Use optimization-based targets as primary. Oracle distillation as comparison baseline — if both produce similar targets, that's evidence the optimization finds meaningful structure rather than artifacts.

#### Information-Theoretic Interpretation

The K embeddings must carry at least I(Q) + I(A) - I(Q;A) bits of information. High mutual information between Q and A means more efficient compression. This suggests a natural curriculum: start with closely related Q-A pairs, increase difficulty by reducing mutual information.

### Training Procedure

#### Phase 1a: Target Generation (Offline)

```python
targets = []

# Fixed mode embeddings (chosen once, used everywhere)
Q_mode = embed(arbitrary_token_1)
A_mode = embed(arbitrary_token_2)

for Q, A in dataset:
    content = initialize_embeddings(size=(K, d))
    
    for step in range(optimization_steps):
        e_q = concat(Q_mode, content)
        e_a = concat(A_mode, content)
        
        loss_q = generator.loss(Q, prompt=e_q)
        loss_a = generator.loss(A, prompt=e_a)
        loss = loss_q + loss_a
        
        content = content - lr * grad(loss, content)
    
    if loss < threshold:
        targets.append((Q, A, content))
```

Use multiple restarts, learning rate schedules, and quality filtering. Expensive but offline.

**The mode mechanism**: The mode position is a simple routing mechanism. Choose two arbitrary tokens from the vocabulary (e.g., "Q" and "A"). Their embeddings remain fixed across all training and inference. During target generation, optimize content positions while holding mode fixed. During inference, set mode to whichever output you want. The mode embedding is *not* optimized — only content positions are. This ensures content encodes shared information while mode purely routes to the appropriate output.

**Deep supervision note** (following HRM/TRM): Consider detaching the computation graph between optimization steps and using intermediate states as additional training signal. This provides richer gradient information than terminal-only supervision.

#### Phase 1a Diagnostics

**Phase 1a is itself a diagnostic** that can validate the approach before proceeding further.

| Observable | What it tells you |
|------------|-------------------|
| **Optimization success rate** | What fraction of pairs converge? Widespread failure suggests the Generator lacks factored representations. |
| **Landscape difficulty** | Do different initializations converge similarly? Forgiving landscapes suggest robust structure. |
| **Distance from M_pretrain** | How far from nearest token embeddings? Large drift means heavy reliance on Generator extrapolation. |
| **Clustering structure** | Do related pairs cluster? Meaningful clustering suggests genuine semantic structure. |
| **Solution heterogeneity** | Qualitatively similar or wildly different? High heterogeneity means complex Diffuser task. |

If Phase 1a produces pathological results, you learn before spending compute on later phases.

#### Phase 1b: Knowledge Backpropagation Diagnostic

A critical question for target generation: when deriving A from Q requires bridging knowledge K, does K get backpropagated into c* even when K is not an explicit optimization target?

**The K Concept**:

```
Q ──────────────────?──────────────────► A
         needs K to bridge
         
K = information required to derive A from Q
K ∉ Q (not in the question)
K ∈ Generator (in the parameters, surfaceable via direct prompting)
K ≠ A (bridging info, not the answer itself)
```

**Concrete example**:
- Q: "What is the largest city in the country that borders Spain to the north?"
- A: "Paris"
- K: "France borders Spain to the north" + "Paris is the largest city in France"

Here K is necessary (can't derive A from Q without it), K is in the Generator (can answer both component questions directly), but K ≠ A (K is the bridging facts, A is just the final answer).

**The Hypothesis**: When we optimize c* for (Q, A) loss, gradients flow backward through the frozen Generator. If the Generator implicitly "knows" K, that knowledge shapes the gradient landscape. The hypothesis: K gets backpropagated into c* — the optimal representation encodes the bridging knowledge even though we never explicitly trained for it.

**Diagnostic Methodology**: Select a small diagnostic set where:

| Criterion | Test |
|-----------|------|
| K is necessary | A is not derivable from Q by surface pattern |
| K is in Generator | Generator produces K components given direct prompts |
| K ≠ A | K is bridging information, not the answer itself |
| Q alone insufficient | Generator performs poorly on A given only Q as prompt |

Then run the ablation:
- Optimize c*_{Q,A} — K implicit, not in loss
- Optimize c*_{Q,A,K} — K explicit, included in loss (with K_mode)
- Compare: representation similarity, A loss achieved

**Interpreting Results**:

| c*_{Q,A} ≈ c*_{Q,A,K}? | A loss similar? | Interpretation |
|------------------------|-----------------|----------------|
| Yes | Yes | K backpropagates; explicit K unnecessary |
| No | Yes | Different encoding achieves same function; interesting |
| No | No, explicit K better | K doesn't backpropagate; must include K |

The third row is the critical failure mode indicating K must be explicitly included for this task type.

**Varying K Complexity**:

| K type | Example | Complexity |
|--------|---------|------------|
| Single fact | "Paris is the capital of France" | Low |
| Multiple facts | Entity relationships requiring composition | Medium |
| Reasoning chain | Multi-step logical derivation | High |
| Procedural | Algorithm execution trace (Sudoku solving steps) | High |

Testing across K complexity reveals the limits of implicit backpropagation. Simple bridging facts may backpropagate reliably; complex reasoning chains may require explicit K.

**Decision Point**:

```
Phase 1a: Optimize c* for (Q,A) pairs
          ↓
    Does optimization converge? Is A loss acceptable?
          ↓
    ┌─────┴─────┐
   Yes          No
    ↓            ↓
Phase 1b:    Add K to loss
K backprop      ↓
diagnostic   Does adding K help?
    ↓            ↓
(validate   ┌────┴────┐
mechanism) Yes        No
            ↓          ↓
    K required    Problem is
    for this      elsewhere
    task type     (Generator lacks K?)
```

This diagnostic can short-circuit Diffuser training: if c* cannot be learned without explicit K, we know before investing in Phase 2.

#### Phase 2: Diffusion Training

Standard diffusion training on the generated targets:

```python
for epoch in range(num_epochs):
    for Q, A, content_star in targets:
        # Forward diffusion: add noise to target
        t = sample_timestep()
        noise = torch.randn_like(content_star)
        alpha_t = noise_schedule(t)
        content_noisy = sqrt(alpha_t) * content_star + sqrt(1 - alpha_t) * noise
        
        # Encode input for conditioning
        input_embeddings = input_encoder(tokenize(Q))
        
        # Diffuser predicts denoised content
        content_pred = diffuser(content_noisy, input_embeddings, t)
        
        # Loss (can be formulated as noise prediction or direct denoising — equivalent)
        loss = mse_loss(content_pred, content_star)
        
        optimizer.step(loss)
```

**On noise level t**: During training, t indexes into the noise schedule, telling the Diffuser how much noise was added. The Diffuser learns to denoise appropriately for each noise level. During inference, t decreases from maximum (pure noise) to zero (clean) across iterative denoising steps.

#### Inference

```python
# Start from pure noise
content = torch.randn(K, d)

# Iterative denoising
for t in reversed(noise_schedule):
    input_embeddings = input_encoder(tokenize(Q))
    content = diffuser.denoise_step(content, input_embeddings, t)

# Generate output
stm = concat(A_mode, content)
response = generator.generate(stm)
```

### Embedding Space Considerations

Understanding where optimized STM embeddings live is important for anticipating failure modes.

#### Two Manifolds

**M_pretrain**: The support of the Generator's pretraining distribution — essentially the discrete token embeddings plus minor perturbations. A sparse set of ~50k points in d-dimensional space.

**M_useful**: The set of embeddings where the Generator produces coherent, low-loss outputs. Defined by the Generator's *generalization* behavior. Strictly larger than M_pretrain.

#### Why This Matters

The target generation process finds embeddings e* that minimize Generator loss. By construction, e* lies on M_useful. The Diffuser learns to output points on M_useful because that's what its training targets are.

**Evidence this works**: Soft prompting research demonstrates that M_useful extends well beyond M_pretrain. Learned continuous prompts often outperform any discrete token sequence.

#### Risks

1. **Heterogeneous solutions**: Different (Q,A) pairs might optimize to qualitatively different regions. The Diffuser must learn a multimodal distribution.

2. **Brittleness**: Optimal points might lie on narrow ridges.

3. **Degenerate solutions**: Optimization might exploit Generator quirks rather than meaningful compression.

#### Mitigations (if needed)

- Regularize toward M_pretrain (penalize distance from nearest token embeddings)
- Semantic equivalence regularization (see above)
- Inspect targets before training; filter pathological cases
- Fine-tune Generator on soft prompts to smooth M_useful

**Recommendation**: Try without regularization first. Inspect what optimal STMs look like. Add constraints only if pathological.

### v1 Evaluation Plan

#### Interpolation

Held-out pairs from training distribution:
- Can Diffuser produce STM enabling correct generation?
- Compare to: raw question as prompt, retrieval-augmented generation

#### Transfer

Different dataset entirely:
- Train on SQuAD, evaluate on TriviaQA
- Transfer success indicates general "compression for generation" rather than dataset-specific patterns

#### Compression/Performance Tradeoff

Vary K, measure:
- Generation quality (loss, accuracy, BLEU, exact match)
- Minimum K achieving baseline performance
- Relationship between required K and Q-A mutual information
- Failure modes when K is too small

#### Interpretability Probes

Phase 1 (before Diffuser training):
- Clustering by topic, question type, answer structure
- Distance distribution from M_pretrain

After Diffuser training:
- Mode separation cleanness
- Attention patterns: which input tokens influence which STM positions
- Intervention: edit STM embeddings, observe output changes

#### Baselines

- Full context: Generator with complete prompt
- Truncation: Naive truncation to K+1 tokens
- Summarization: LLM-generated summary compressed to similar size
- Random soft prompt: Same architecture, random targets
- Oracle distillation: Targets from full-context model

### v1 Open Questions

**Addressable during Phase 1**:
- Optimization landscape difficulty
- M_useful characterization
- Minimum viable K per task type
- Effectiveness of semantic equivalence regularization
- K backpropagation: for what complexity of bridging knowledge does implicit encoding via gradient flow succeed vs require explicit K targets?

**Requires Diffuser training**:
- Transfer across datasets
- Optimal Diffuser architecture (depth, width, attention layers)
- Number of diffusion steps needed at inference

---

## Part IV: v2+ — Full Architecture

v2 extends the validated v1 approach with long-term memory and streaming input.

### Additional Components

#### Long-Term Memory (LTM)

LTM stores lossy compressions of prior STM states for later retrieval.

- **Storage**: Embeddings, not tokens
- **Structure**: Each entry can carry prediction metadata:
  - **Content**: The compressed STM state
  - **Effect**: Predicted future utility (what will this memory enable?)
  - **Timing**: Conditions under which to verify the prediction

**Note on memory structure**: Not every memory needs (effect, timing) predictions. Some memories are observations of structure — patterns noticed, facts encoded — without explicit utility predictions. The (effect, timing) metadata is what the Gating Model *learns to attach* when useful, not an intrinsic requirement. Task pressure shapes whether and how this structure emerges.

#### STM Encoder

Distinct from the Input Encoder:

| | Input Encoder | STM Encoder |
|-|---------------|-------------|
| **Input** | Tokens | STM embedding buffer |
| **Output** | Sequence of embeddings | Single embedding vector |
| **Purpose** | Condition Diffuser | Query LTM, input to Gating Model |
| **Version** | v1+ | v2+ |

#### Gating Model

Controls what gets committed to LTM:

| Decision | Effect |
|----------|--------|
| **Discard** | Not worth persisting |
| **Accumulate** | Average into buffer being built |
| **Commit buffer** | Push accumulated representation to LTM |
| **Commit solo** | This single state gets its own LTM entry |
| **Commit after complete** | Wait until processing completes; captures final integrated state when intermediates have no retrieval value |

Variable compression emerges from these decisions:
- Low-valence sequences → aggressive accumulation → coarse memories
- High-valence moments → solo commits → high-fidelity memories

### LTM Read Path

```python
stm_embedding = stm_encoder(STM)
retrievals = query_ltm(stm_embedding, top_N)  # Similarity search
```

Retrieved embeddings condition the Diffuser alongside input embeddings. The Diffuser learns context-dependent integration — the same LTM embedding may influence STM differently depending on current state, analogous to how human recall surfaces different details depending on current goals.

### LTM Write Path

The Gating Model observes each STM state and decides whether/how to persist it.

**Memory structure on write**: Each entry carries the content embedding plus optional (effect, timing) predictions for learning signal.

### Gating Model Training Signal

This is the hardest problem in the full architecture.

#### The Credit Assignment Problem

```
commit decision (now)
    → memory exists in LTM
        → memory gets retrieved (later, maybe)
            → retrieval affects STM
                → STM affects response quality
                    → compare to predicted effect
```

Difficulties:
1. **Temporal distance**: Gradient must flow through many steps
2. **Sparse activation**: Unretrieved memories provide no signal

#### Potential Approaches

| Approach | Description | Tradeoff |
|----------|-------------|----------|
| **Retrieval prediction** | Predict whether memory will be retrieved | Shorter horizon, but retrieval ≠ usefulness |
| **Reconstruction loss** | Penalize information loss | Encourages fidelity, not utility |
| **Contrastive value** | Compare quality with vs. without memories | Direct signal, expensive counterfactuals |
| **Hindsight relabeling** | Relabel which commits should have happened | Uses future info, training only |
| **Imitation from oracle** | Full-context model labels importance | Sidesteps credit assignment, requires oracle |

The right approach likely combines multiple signals. This is the critical research problem for v2+.

### STM Encoder Dual-Purpose Tension

The STM Encoder serves two purposes:
1. **Retrieval**: Query embedding for LTM similarity search
2. **Gating**: Input for commit decisions

These objectives may conflict. What makes a good retrieval query may differ from what's informative for commit decisions.

**Potential solution**: Shared encoder with distinct projection heads — a retrieval head optimized for query similarity, a gating head for commit decisions. This preserves shared representations while allowing task-specific adaptation.

### Streaming Input Processing

v2+ handles streaming (token-by-token) input:

```python
for token in input_stream:
    stm_embedding = stm_encoder(STM)
    retrievals = query_ltm(stm_embedding, top_N)
    STM = diffuser(STM, token, retrievals)  # Incremental update
    
    # Gating decision
    decision = gating_model(stm_embedding)
    apply_decision(decision, STM, LTM)

response = generator(STM)
```

**Open questions for streaming**:
- How many diffusion steps per incoming token?
- How does partial input affect the diffusion process?
- Does Generator output flow back through the loop for multi-turn?

### v2+ Information Flow

```
                         User input tokens (streaming)
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────┐
         │                                                  │
         │    ┌─────────┐                                   │
         │    │   STM   │◄──────────────────────┐           │
         │    │(embeds) │                       │           │
         │    └────┬────┘                       │           │
         │         │                            │           │
         │         ▼                            │           │
         │    ┌─────────┐     query        ┌────┴────┐      │
         │    │   STM   │─────────────────►│   LTM   │      │
         │    │ Encoder │                  │(embeds) │      │
         │    └────┬────┘                  └────┬────┘      │
         │         │                            │           │
         │         │ embedding                  │ retrieved │
         │         │                            │ embeddings│
         │         ▼                            │           │
         │    ┌─────────┐                       │           │
         │    │ Gating  │                       │           │
         │    │ Model   │                       │           │
         │    └────┬────┘                       │           │
         │         │                            │           │
         │         │ write decision             │           │
         │         ▼                            │           │
         │    ┌─────────┐                       │           │
         │    │   LTM   │◄──(commit)            │           │
         │    └─────────┘                       │           │
         │                                      │           │
         │              ┌───────────────────────┘           │
         │              │                                   │
         │              ▼                                   │
         │    ┌───────────────────┐     cross-attn         │
         │    │     Diffuser      │◄────(input token)       │
         │    │                   │                         │
         │    │(STM, LTM, input)  │─────────────────────────┘
         │    │      → STM'       │
         │    └───────────────────┘
         │
         └──────── loop for each input token ───────────────
                                    │
                                    ▼ (input complete)
                            ┌─────────────┐
                            │  Generator  │
                            │ (STM→resp)  │
                            └──────┬──────┘
                                   │
                                   ▼
                              Response
```

### v2+ Open Questions

- LTM capacity limits and eviction policy
- Gating model training signal (the hard problem)
- Streaming: diffusion steps per token, partial input handling
- Multi-turn: does Generator output flow back?
- Cold start: initial LTM contents
- STM Encoder dual-purpose optimization dynamics

---

## Part V: Design Principles

### Core Principles

1. **Context is instrumental**: No "good context" independent of downstream task performance. Target distribution defined operationally as "embeddings that minimize generation loss."

2. **Iterative refinement over one-shot**: Context building is diffusion-like — condition, denoise, converge. Not a single retrieval decision.

3. **Compression through optimization**: Capacity constraints force efficient encodings that preserve task-relevant information while discarding noise.

4. **Generator-legibility as interpretability constraint**: STM must encode information the Generator can use, forcing structured, "readable" representations.

5. **Memory as prediction** *(v2+)*: Memories can encode predictions about future utility (effect, timing), providing learning signal beyond terminal reward.

6. **Structure through learning** *(v2+)*: Memory embeddings develop task-relevant structure via optimization pressure, not imposed schema.

### Design Heuristic

This architecture is guided by introspection on human memory, used loosely as design intuition rather than strong isomorphism:
- Working memory as a compressed, task-relevant representation
- Long-term memory as lossy compression of working memory states
- Retrieval as context-dependent decompression
- Relevance defined by anticipated utility, not just content similarity
- Prediction error as a primary driver of memory refinement

---

*Document synthesized from design sessions, January 2026*
*v0.6: Major restructure — v1/v2 separation, improved section ordering, distinguished Input Encoder from STM Encoder, added semantic equivalence regularization, HRM/TRM references, systematic K selection notes*
*v0.6.1: Added Phase 1b Knowledge Backpropagation Diagnostic — testing whether bridging knowledge K gets implicitly encoded in c* via gradient flow*
*v0.6.2: Restructured Part III — Training Procedure now contains Phase 1a, 1a Diagnostics, Phase 1b, and Phase 2 in sequence; Embedding Space Considerations moved after training; mode mechanism explanation moved to Training Procedure where it's used*