# Bridging the teacher forcing gap in soft prompt optimization

**The core issue you're experiencing—low cross-entropy loss but degenerate autoregressive outputs—is definitively exposure bias**, a well-documented phenomenon where training with teacher forcing creates models that can't handle their own prediction errors at inference. For soft prompt optimization against frozen LLMs, this problem is especially severe because the prompts overfit to conditioning on ground-truth contexts they'll never see during generation. The solution requires combining training modifications (scheduled sampling, auxiliary losses) with inference interventions (constrained decoding, repetition penalties), implemented through a curriculum that gradually shifts from teacher forcing toward autoregressive behavior.

## Scheduled sampling adapts the training distribution to match inference

The foundational technique for exposure bias is **scheduled sampling** (Bengio et al., NeurIPS 2015), which probabilistically mixes ground-truth and model-generated tokens during training. At each position, a coin flip with probability ε determines whether to use the target token or the model's own prediction as context for the next step. The key insight: ε starts at **1.0** (pure teacher forcing) and decays over training toward **0.2-0.3**, gradually exposing the model to its own mistakes.

Three validated decay schedules exist. **Linear decay** (ε = k - ci) provides simple control. **Exponential decay** (ε = k^i) drops quickly then plateaus. The **inverse sigmoid schedule** (ε = k/(k + exp(i/k)), typically k=15-25) is recommended for smooth transitions. Results on image captioning showed BLEU-4 improvements from **28.8 → 30.6** using inverse sigmoid decay.

For **Transformers specifically**, Mihaylova & Martins (ACL 2019) developed a two-pass approach: the first pass runs teacher-forced generation to collect predictions, then the second pass trains on mixed sequences. This preserves Transformer parallelism while enabling scheduled sampling. The **Parallel Scheduled Sampling** variant (ICLR 2020) generates entire sequences in parallel, then randomly substitutes ground-truth tokens before computing loss.

Adapting this for frozen LLMs requires a critical modification: operate at the **embedding level** rather than discrete tokens. Instead of choosing between ground-truth or predicted tokens, interpolate their embeddings:

```python
mixed_embedding[t] = ε * embed(y_gt[t-1]) + (1-ε) * embed(ŷ[t-1])
```

This keeps gradients flowing to soft prompts while simulating autoregressive exposure. Temperature-annealed **Gumbel-Softmax** (Jang et al., ICLR 2017) offers a more principled approach—create differentiable "soft tokens" as weighted combinations of the embedding matrix using the predicted distribution, enabling end-to-end gradient flow through the generation trajectory.

## Differentiable decoding enables direct optimization of generation quality

Standard training can't optimize generation quality because argmax sampling breaks gradient flow. **Gumbel-Softmax reparameterization** solves this by replacing discrete sampling with a differentiable relaxation:

$$y_i = \text{softmax}\left(\frac{\log \pi_i + g_i}{\tau}\right)$$

where g_i ~ Gumbel(0,1) and τ controls temperature. High τ produces soft distributions; low τ approximates hard sampling. **Straight-through Gumbel-Softmax** uses hard samples in the forward pass but soft gradients in the backward pass, combining discrete outputs with continuous gradient estimation.

For soft prompt optimization against frozen LLMs, implement this as:

```python
def forward_with_soft_decoding(soft_prompt, embedding_matrix, num_tokens, τ=1.0):
    current_input = soft_prompt
    for _ in range(num_tokens):
        logits = frozen_llm.forward_embeddings(current_input)[-1]  # Last position
        soft_probs = F.gumbel_softmax(logits, tau=τ, hard=False)
        soft_token = soft_probs @ embedding_matrix  # Weighted embedding
        current_input = torch.cat([current_input, soft_token.unsqueeze(0)], dim=0)
    return current_input
```

**Temperature annealing** is critical: start with τ=1.0-2.0 and anneal toward τ=0.5 over training. This allows early exploration while later stages approximate discrete behavior.

When differentiable approaches face limitations (frozen model only accepts discrete tokens), **REINFORCE with baseline subtraction** provides unbiased gradient estimates. The RLOO (REINFORCE Leave-One-Out) variant from recent LLM alignment work (Ahmadian et al., ACL 2024) uses K samples where each sample's baseline is the average reward of the other K-1 samples, significantly reducing variance without learned critics.

## Auxiliary losses prevent degeneration by constraining representation space

Beyond modifying how training happens, adding **auxiliary losses** directly penalizes degenerate patterns. Two approaches stand out as highly effective and practical.

**Unlikelihood training** (Welleck et al., ICLR 2020) explicitly decreases the probability of repeated tokens:

$$\mathcal{L}_{\text{UL}} = -\sum_{c \in \mathcal{C}_t} \log(1 - p_\theta(c | x_{<t}))$$

where C_t contains tokens appearing earlier in the sequence. This reduces sequence-level repetition (seq-rep-4) from **~50% to ~4%** while maintaining perplexity. For soft prompt optimization, apply this to teacher-forced logits with weight α=0.5-1.0.

**SimCTG contrastive loss** (Su et al., NeurIPS 2022) addresses the root cause of degeneration—anisotropic token representations with cosine similarities around **0.95**. The contrastive objective pushes representations apart:

$$\mathcal{L}_{\text{CL}} = \frac{1}{|x|(|x|-1)} \sum_i \sum_{j \neq i} \max\{0, \rho - s(h_i, h_i) + s(h_i, h_j)\}$$

with margin ρ=0.5. Combined with **contrastive search decoding** at inference—which balances model confidence against similarity to previous tokens—this approach dramatically reduces repetition while maintaining fluency.

The recommended combined objective for soft prompt optimization:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MLE}} + 0.5 \cdot \mathcal{L}_{\text{UL}} + 0.3 \cdot \mathcal{L}_{\text{CL}} + 0.2 \cdot \mathcal{L}_{\text{AR}}$$

where L_AR is a periodic autoregressive validation loss computed on actual greedy generations.

## The Vec2Text architecture solves exact inversion through iterative correction

The **Vec2Text framework** (Morris et al., ICLR 2024) is directly relevant to your prompt inversion problem and achieves **92% exact reconstruction** for 32-token texts. The key insight: train a correction model that explicitly conditions on model-generated hypotheses, addressing the train-test mismatch head-on.

The architecture uses multi-step inversion: (1) a "zero-step" model generates an initial hypothesis from the embedding, (2) a correction model iteratively refines the hypothesis by conditioning on (target_embedding, current_hypothesis, hypothesis_embedding) tuples. This explicitly trains on the model's own generations rather than only ground-truth contexts.

For your setting, adapt this as **iterative soft prompt refinement**:
1. Train initial soft prompts with standard CE loss
2. Generate text autoregressively with current prompts  
3. Train a "correction" stage that takes (target, generated, soft_prompt) and produces updated prompts
4. Iterate until convergence

This approach naturally handles the train-test gap by making the correction model robust to generation errors.

## MLP reparameterization and regularization improve prompt generalization

**Prefix-Tuning** (Li & Liang, ACL 2021) discovered that directly optimizing soft prompt embeddings performs significantly worse than optimizing through an MLP reparameterization:

```python
prefix_mlp = nn.Sequential(nn.Linear(d, 4*d), nn.Tanh(), nn.Linear(4*d, d))
soft_prompt = prefix_mlp(learnable_parameters)  # Indirect parameterization
```

The MLP is discarded after training—only the output embeddings are kept. This constrains the optimization landscape and consistently improves stability and generalization.

Additional regularization strategies from the literature include **perplexity regularization** (penalize prompts that would be "surprising" to the LM), **embedding-space constraints** (encourage prompts to be convex combinations of existing token embeddings), and **gradient alignment** (ProGrad: update prompts only when gradients align with "general" directions). For exact reconstruction, constraining soft prompts toward the embedding manifold may prevent overfitting to teacher-forcing dynamics.

## Inference-time strategies provide immediate improvements without retraining

While training modifications address root causes, **inference-time interventions** offer immediate improvements for existing prompts.

**Constrained beam search** can force specific token sequences during generation. NeuroLogic Decoding (Lu et al., NAACL 2021) handles arbitrary lexical constraints expressible in predicate logic with runtime equivalent to conventional beam search. For HuggingFace models:

```python
from transformers import PhrasalConstraint
outputs = model.generate(
    input_ids,
    constraints=[PhrasalConstraint(tokenizer(target_text).input_ids)],
    num_beams=10,
    length_penalty=0.8
)
```

For anti-degeneration without hard constraints, combine **repetition penalty** (1.1-1.3 for short sequences), **n-gram blocking** (no_repeat_ngram_size=2), and **nucleus sampling** (top_p=0.9). The **LZ penalty** (2025) applies information-theoretic repetition penalties based on Lempel-Ziv compression codelengths, enabling greedy decoding without degenerate repetition.

**Contrastive search** (from SimCTG) balances model confidence against degeneration at each step:

$$x_t = \arg\max_{v \in V^{(k)}} \left\{(1-\alpha) \cdot p(v|x_{<t}) - \alpha \cdot \max_j s(h_v, h_{x_j})\right\}$$

with k=5-10 candidates and α=0.5-0.8 degeneration penalty.

## A practical four-stage curriculum for soft prompt optimization

Based on the research, implement training in four stages with progressively decreasing teacher forcing:

**Stage 1 (0-20% training): Pure teacher forcing on short sequences**
- Use standard CE loss with full teacher forcing (ε=1.0)
- Limit sequence length to 25% of maximum (e.g., 8-12 tokens for 50-token targets)
- Apply MLP reparameterization and embedding noise (σ=0.01)
- Goal: Learn basic token distributions and stable initialization

**Stage 2 (20-60% training): Gradual scheduled sampling**
- Decay ε from 0.9 to 0.5 using inverse sigmoid schedule
- Increase sequence length to 50%, then 75% of maximum
- Add unlikelihood loss (α=0.5) and SimCTG contrastive loss (β=0.2)
- Implement embedding-level interpolation between GT and predicted tokens

**Stage 3 (60-80% training): Autoregressive-heavy training**  
- Continue decaying ε from 0.5 to 0.2
- Full sequence length
- Add periodic autoregressive validation loss (every 10-20 batches)
- Use Gumbel-Softmax with temperature annealing (τ: 1.0 → 0.5)

**Stage 4 (80-100% training): Mixed objective fine-tuning**
- Combine CE loss with sequence-level REINFORCE using generation quality reward
- Weight: 0.7 * L_CE + 0.2 * L_AR + 0.1 * L_REINFORCE
- Apply contrastive search during validation to monitor true generation quality
- Optional: Early stopping based on autoregressive performance, not CE loss

## Implementation checklist with hyperparameters

| Component | Recommendation | Source |
|-----------|---------------|--------|
| Optimizer | AdamW, weight decay 0.01 | Lester et al. |
| Learning rate | 2e-4 to 3e-2 (higher than fine-tuning) | Prefix-Tuning |
| Warmup | 10-20% of training steps | Vec2Text |
| Prompt length | 16-64 tokens; shorter may generalize better | Empirical |
| Initialization | Semantically relevant token embeddings | Lester et al. |
| Scheduled sampling decay | Inverse sigmoid, k=15-25 | Bengio et al. |
| Minimum ε | 0.2 (never go to pure autoregressive) | Empirical |
| Gumbel temperature | Anneal 1.0 → 0.5 | Standard practice |
| Unlikelihood weight | α = 0.5-1.0 | Welleck et al. |
| Contrastive weight | β = 0.1-0.3 | SimCTG |
| Repetition penalty (inference) | 1.1-1.3 | CTRL |
| Contrastive search α | 0.5-0.8 | SimCTG |

## Conclusion

The teacher forcing gap in soft prompt optimization stems from a fundamental distribution mismatch: prompts optimized for ground-truth contexts fail when faced with model-generated (potentially erroneous) contexts at inference. The most effective solution combines **scheduled sampling** during training to gradually expose prompts to autoregressive dynamics, **auxiliary losses** (unlikelihood, contrastive) to prevent degeneration, and **constrained decoding** at inference to guide generation toward target sequences.

For exact text reconstruction, the **Vec2Text iterative correction** architecture offers a promising direction—explicitly training on model generations rather than only teacher-forced contexts. MLP **reparameterization** from Prefix-Tuning and **embedding-space regularization** improve prompt generalization by constraining optimization to well-behaved regions.

The key insight across all approaches: **validation should match inference**. Monitor greedy/beam search generation quality throughout training, not just teacher-forced cross-entropy. A prompt achieving 200x lower CE loss but producing degenerate outputs is overtrained for the wrong objective—the fix is changing what you optimize, not optimizing harder.