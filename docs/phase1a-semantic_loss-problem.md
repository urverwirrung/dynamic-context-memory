# Repetition and Stopping in Soft Prompt Optimization

## Problem Statement

### Context

We are optimizing soft prompt embeddings (continuous vectors) to make a frozen language model reconstruct target sequences. The optimization objective is to find embeddings `c` such that `decode(LM, c) = target`.

**Recent progress:** We implemented a semantic similarity loss that successfully addresses the "semantic equivalent penalty" problem. The model now correctly generates targets like "Greater London is divided into what two groups of boroughs?" that it previously couldn't learn.

### The New Problem: Repetition After Target

With semantic loss, optimization often finds embeddings that produce the correct target—but the model doesn't stop. It continues generating, typically repeating the target or degenerating into repetitive patterns.

**Observed failure modes:**

```
Target:    "ATSC"
Generated: "ATSCATSCATSCAT"

Target:    "Place Broglie"
Generated: "Place BrogliePlace BrogliePlace"

Target:    "Inner London and Outer London"
Generated: "Inner London and Outer London Inner London and Outer London"

Target:    "Greater London is divided into what two groups of boroughs?"
Generated: "Greater London is divided into what two groups of boroughs? Wait, no, that"
```

**Degeneration case:**
```
Target:    "Where is the Opera House located?"
Generated: "Where is the622222222"
```

### Why This Happens

1. **No explicit stopping signal**: The loss function rewards matching the target tokens but doesn't penalize extra tokens after the target.

2. **Semantic loss is permissive**: Cosine similarity between "ATSC" and "ATSCATSCATSCAT" hidden states may still be high because the target content is present.

3. **Autoregressive momentum**: Once the model starts a pattern, it tends to continue—especially short patterns that create strong local context.

4. **No EOS learning**: We're optimizing embeddings to produce content, not to produce content + stop. The model has no reason to emit EOS after the target.

### Current Results

| Target | Generated | Status |
|--------|-----------|--------|
| "Greater London is divided into what two groups of boroughs?" | Same + "Wait, no, that" | ✓ Correct prefix |
| "What did the Grand Alliance propose as the new standard for SDTV and HDTV?" | Exact match | ✓ Perfect |
| "Where is the Opera House located?" | "Where is the622222222" | ✗ Degeneration |
| "Inner London and Outer London" | Same × 2 | ✓ Correct but repeats |
| "Place Broglie" | Same × 2.5 | ✓ Correct but repeats |
| "ATSC" | Same × 3.5 | ✓ Correct but repeats |

**Convergence rate: 83% (5/6)** — but repetition is present in most "successful" cases.

### Desired Properties of a Solution

1. **Target completeness**: Generate all target tokens
2. **Clean stopping**: Stop after the target (or emit EOS)
3. **No repetition**: Don't repeat the target or parts of it
4. **No degeneration**: Don't collapse into repetitive garbage (e.g., "622222222")

### Potential Approaches

#### 1. Length-aware loss
Penalize outputs longer than target:
```python
length_penalty = max(0, len(generated) - len(target)) * penalty_weight
loss = semantic_loss + length_penalty
```
**Problem**: Not differentiable through discrete token generation.

#### 2. EOS prediction bonus
Add term rewarding EOS prediction at position `len(target)`:
```python
eos_bonus = log_prob(EOS | context_at_target_end)
loss = semantic_loss - eos_weight * eos_bonus
```

#### 3. Repetition penalty in loss (not just inference)
Current repetition penalty is only applied during greedy decoding. Could add unlikelihood loss during training:
```python
# Penalize probability of tokens that already appeared
for token in seen_tokens:
    loss += alpha * log(1 - prob(token))
```

#### 4. Truncated evaluation
Only evaluate the first `len(target)` tokens of generation:
```python
generated_truncated = generated[:len(target)]
loss = semantic_loss(generated_truncated, target)
```
**Problem**: Doesn't teach the model to stop—just ignores the problem.

#### 5. Contrastive loss against repetitions
Explicitly push embeddings away from generating repetitions:
```python
# Generate with current embedding
# If repetition detected, create negative example
loss = semantic_loss - margin_loss(embedding, repetitive_output)
```

#### 6. Two-phase optimization
1. First optimize for target content (current approach)
2. Then fine-tune for clean stopping (add EOS constraint)

### Questions for Prior Art Search

1. **Controlling generation length** in soft prompt / prefix tuning
2. **Anti-repetition losses** for training (not just inference-time penalties)
3. **Learning to stop** — teaching models when to emit EOS
4. **Repetition in autoregressive models** — causes and training-time solutions
5. **Unlikelihood training** for repetition prevention
6. **Length control** in text generation
7. **Curriculum learning** for generation — content first, then style/length

### Related Problems

- **Exposure bias**: Model sees ground truth during training but own predictions at inference (we addressed this with scheduled sampling, may need to revisit)
- **Neural text degeneration**: "The Curious Case of Neural Text Degeneration" (Holtzman et al.) — repetition is a known failure mode
- **Repetition in summarization/translation**: Similar issues in seq2seq tasks
- **Mode collapse**: Model finds simple repetitive patterns that minimize loss

### Success Criteria

1. Generated output matches target exactly (or with EOS)
2. No tokens generated after target (or only EOS)
3. No repetitive patterns within or after target
4. Maintains current convergence rate (83%+) while fixing repetition

### Experiment Log

**2024-01-17: Semantic loss implementation**
- Implemented token-level + sentence-level semantic similarity loss
- Solved the "Greater London" problem (semantic equivalents now learnable)
- New problem emerged: repetition after target
- Convergence: 83% (5/6), but most have repetition artifacts
