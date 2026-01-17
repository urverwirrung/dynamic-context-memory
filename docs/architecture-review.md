# MMS Review: dcm-plan.md

## Summary
- **Lenses applied**: accuracy, completeness, consistency
- **Overall agreement**: 0.72
- **Issues found**: 31 (1 critical, 10 high, 15 medium, 5 low)

| Lens | Issues | Model Agreement | Declaration |
|------|--------|-----------------|-------------|
| accuracy | 10 | 0.69 | improvements_possible |
| completeness | 17 | 0.73 | improvements_possible |
| consistency | 10 | 0.71 | improvements_possible |

---

## Critical Issues

### [consistency-001] Variable name collision: 'K' overloaded
**Confidence**: 0.95 (both agree)

In Phase 1b, 'K' denotes bridging knowledge ("K = information required to derive A from Q"). Throughout the rest of the document, 'K' denotes the number of content embedding positions in STM ("K content positions", "shape (K, d)").

**Location**: Part III > Phase 1b vs entire document
**Suggestion**: Rename bridging knowledge to 'B' or 'K_bridge' to disambiguate from K_content.
    **Suggestion Approved: Rename bridging knowledge to 'B'**

---

## High Priority Issues

### [accuracy-001] Information-theoretic bound conflates dimensions with bits
**Confidence**: 0.95 (both agree) | **Severity**: high

The claim "K × d bits must encode at least I(Q) + I(A) - I(Q;A)" incorrectly equates embedding dimensions with information-theoretic bit capacity. K×d counts real-valued dimensions, not bits.

**Location**: Part III > STM Structure for v1, Note on K selection
**Suggestion**: Reframe as: "The K embeddings must have sufficient capacity to encode at least H(Q,A) bits, where capacity depends on K, d, and effective precision."
    **Issue Recognized: Resolution deferred to planning session**
---

### [accuracy-002] HRM/TRM terms used without citation
**Confidence**: 0.85 (both agree) | **Severity**: medium

"Hopfield Retrieval Memory (HRM)" and "Transformer Retrieval Memory (TRM)" are referenced without citation or definition. These are not standard terms in the literature.

**Location**: Part I > Relationship to Prior Work
**Suggestion**: Add citations or clarify if these are author-coined terms.
    **Suggestion Approved: retrieve and add citations; HRM stand for 'Hierarchical Reasoning Model", and TRM for 'Tiny Recursive Model'**
---

### [completeness-001] No success/failure criteria defined
**Confidence**: 0.95 (both agree) | **Severity**: critical→high

The evaluation plan lacks quantitative acceptance criteria. No thresholds for optimization success rate, minimum accuracy on SQuAD, or go/no-go decisions.

**Location**: Part III > v1 Evaluation Plan
**Suggestion**: Define explicit metrics: "Phase 1a succeeds if optimization converges for >80% of examples with loss < X."
    **Issue Recognized: Resolution deferred to planning session**
---

### [completeness-002] No handling of adversarial/malicious inputs
**Confidence**: 0.90 (both agree) | **Severity**: high

No discussion of soft prompt injection, memory poisoning, or adversarial inputs exploiting the diffusion process.

**Location**: Missing section
**Suggestion**: Add security/safety section with threat model and mitigations.
    **no lol. we really need to think through what went wrong for this kind of suggestion to come up in this context**

---

### [completeness-003] No computational cost analysis
**Confidence**: 0.90 (both agree) | **Severity**: high

No estimates for Phase 1a compute (steps per example, total FLOPs), training costs, or memory requirements.

**Location**: Missing section
**Suggestion**: Include concrete compute estimates and scaling analysis.
    **no. we really need to think through what went wrong for this kind of suggestion to come up in this context**

---

### [completeness-004] No privacy/governance for LTM
**Confidence**: 0.85 (Codex) | **Severity**: high

No mention of data retention policies, privacy considerations, or user controls for LTM storage.

**Location**: Part IV > LTM
**Suggestion**: Add stakeholder considerations for privacy, data minimization, and memory provenance.
    **what??? we really need to think through what went wrong for this kind of suggestion to come up in this context**
---

### [completeness-005] Unstated assumption: Generator knowledge sufficiency
**Confidence**: 0.85 (Claude) | **Severity**: high

Assumes the frozen Generator has sufficient implicit knowledge to support factored representations. If Generator knowledge is incomplete, optimization fails silently.

**Location**: Part III > Target Generation
**Suggestion**: Add diagnostics for detecting Generator knowledge gaps.
    **failure is not silent, it will show in the high loss on A of the best representation - ie: the A mode generation of the best c representation will not be an adequate response to the question. that is the actual threshold.

---

### [completeness-006] No handling for very long inputs
**Confidence**: 0.85 (both agree) | **Severity**: high

No discussion of input length limits or chunking strategies when inputs exceed Input Encoder capacity.

**Location**: Part III > v1 Architecture
**Suggestion**: Document maximum input length and hierarchical strategies.

---

### [completeness-007] No inference-time confidence estimation
**Confidence**: 0.85 (both agree) | **Severity**: high

No mechanism to detect when the system is producing unreliable outputs at inference time.

**Location**: Part III > Inference
**Suggestion**: Add uncertainty quantification or confidence mechanisms.

---

### [consistency-002] Frozen Generator vs fine-tuning conflict
**Confidence**: 0.90 (both agree) | **Severity**: medium

v1 scope states "Frozen pretrained Generator" but Embedding Space Considerations suggests "Fine-tune Generator on soft prompts" as a mitigation.

**Location**: Part III > v1 Scope vs Embedding Space Considerations
**Suggestion**: Mark fine-tuning as v2+ or optional ablation, not v1 scope.

---

## Medium Priority Issues

### [accuracy-003] Geographic error in Phase 1b example
**Confidence**: 0.85 (Claude only) | **Severity**: medium

"France borders Spain to the north" is geographically incorrect from Spain's perspective. France is north of Spain, so France borders Spain to the south.

**Location**: Part III > Phase 1b
**Suggestion**: Change to "France is north of Spain" or "Spain's northern border is with France."

---

### [accuracy-004] Soft prompt claims overstated
**Confidence**: 0.80 (both agree) | **Severity**: medium

"Learned continuous prompts often outperform any discrete token sequence" is too strong without exhaustive search evidence.

**Location**: Part III > Embedding Space Considerations
**Suggestion**: Qualify as "can outperform hand-crafted or limited-search discrete prompts."

---

### [accuracy-005] "Provably minimize" is incorrect
**Confidence**: 0.90 (both agree) | **Severity**: medium

Optimization is nonconvex; practical procedures find local/approximate minima, not global optimality.

**Location**: Part III > Oracle Distillation table
**Suggestion**: Use "optimized to locally minimize" or "approximately minimize."

---

### [consistency-003] STM shape notation inconsistency
**Confidence**: 0.85 (Claude only) | **Severity**: medium

Part II uses "(N, d)" for STM shape; Part III uses "(K+1, d)". The relationship between N and K+1 is not clarified.

**Location**: Part II vs Part III
**Suggestion**: State explicitly that N = K+1 in v1, or use consistent notation.

---

### [consistency-004] Logic gap: Joint optimization → factored representations
**Confidence**: 0.95 (both agree) | **Severity**: medium

The claim that successful joint Q-A optimization proves the Generator has "factored representations" has a missing step. Success could also reflect memorization or surface feature correlation.

**Location**: Part III > Why Joint Q-A Optimization
**Suggestion**: Add criteria distinguishing genuine semantic factorization from alternative explanations.

---

### [consistency-005] Logic gap: Inference generalization unstated
**Confidence**: 0.90 (both agree) | **Severity**: medium

Training produces targets for specific (Q,A) pairs. Inference assumes the Diffuser generalizes to new Q. This assumption about smooth Q→content mapping is unstated.

**Location**: Part III > Inference
**Suggestion**: Make the generalization assumption explicit with supporting rationale.

---

### [consistency-006] Logic gap: v1 iterative denoising → v2 per-token updates
**Confidence**: 0.85 (both agree) | **Severity**: medium

v1 trains with noise-based diffusion steps on complete input. v2+ streaming shows single diffuser call per token. The architectural bridge is missing.

**Location**: Part III vs Part IV
**Suggestion**: Explain how the same Diffuser adapts from iterative denoising to incremental token updates.

---

### [completeness-008] No STM slot semantics defined
**Confidence**: 0.60 (Codex only) | **Severity**: medium

No explicit definition of whether STM slots are ordered, interchangeable, or have learned roles.

**Location**: Part II > STM
**Suggestion**: Specify positional semantics and whether Diffuser enforces any structure.

---

### [completeness-009] No hyperparameter sensitivity analysis
**Confidence**: 0.80 (both agree) | **Severity**: medium

No discussion of which hyperparameters (K, noise schedule, learning rates) are critical or their expected ranges.

**Location**: Part III
**Suggestion**: Add guidance on hyperparameter selection and sensitivity.

---

### [completeness-010] Generator architecture requirements unspecified
**Confidence**: 0.85 (Claude only) | **Severity**: medium

No specification of which Generator architectures, sizes, or pretraining regimes are expected to work.

**Location**: Part III
**Suggestion**: Document minimum model size, architecture constraints, and pretraining objectives.

---

## Low Priority Issues

### [accuracy-006] Vocabulary size approximation varies
**Confidence**: 0.75 (both agree) | **Severity**: low

"~50k points" varies significantly across models (32k-100k+).

**Location**: Part III > Embedding Space Considerations
**Suggestion**: Change to "tens of thousands" or specify model-dependence.

---

### [accuracy-007] "Natural questions" may need capitalization
**Confidence**: 0.70 (Claude only) | **Severity**: low

If referring to Google's Natural Questions dataset, should be capitalized.

**Location**: Part III > v1 Scope
**Suggestion**: Capitalize as "Natural Questions" if referring to the dataset.

---

### [accuracy-008] Noise prediction vs direct denoising not strictly equivalent
**Confidence**: 0.75 (both agree) | **Severity**: low

They have different training dynamics in practice; oversimplified here.

**Location**: Part III > Phase 2: Diffusion Training
**Suggestion**: Note equivalence is "up to reparameterization" with potentially different dynamics.

---

### [consistency-007] Gating Model decisions differ between Part II (4) and Part IV (5)
**Confidence**: 0.80 (both agree) | **Severity**: low

Part IV adds "Commit after complete" not mentioned in Part II overview.

**Location**: Part II vs Part IV
**Suggestion**: Update Part II to note "additional decisions in v2+" or include all five.

---

### [completeness-011] Multi-modal scope not explicit
**Confidence**: 0.70 (both agree) | **Severity**: low

Document is implicitly text-only; should be stated explicitly.

**Location**: Part III > v1 Scope
**Suggestion**: State "v1 scope is text-only" explicitly.

---

## Coverage Assessment

### Topics Well Covered
- Core diffusion analogy and conditioning mechanism
- v1/v2+ component boundaries and responsibilities
- Target generation rationale and diagnostics
- Phase 1a/1b diagnostic framework
- Cross-attention conditioning mechanism
- Embedding space considerations (M_pretrain vs M_useful)
- Gating model credit assignment problem

### Topics Undercovered
- Quantitative success criteria and go/no-go thresholds
- Computational cost and scaling analysis
- Safety, privacy, and governance of LTM
- Adversarial robustness and failure modes
- Generator architecture requirements
- Streaming transfer from v1 to v2+
- Inference-time uncertainty/confidence

### Overall Completeness: **0.72**

---

## Model Agreement Analysis

| Category | Both Agree | Claude Only | Codex Only |
|----------|------------|-------------|------------|
| Accuracy issues | 7 | 2 | 1 |
| Completeness gaps | 12 | 5 | 5 |
| Consistency issues | 7 | 2 | 1 |

**High-confidence findings** (both models agree): K-bits conflation, K variable collision, no success criteria, adversarial robustness gap, compute cost gap, joint optimization logic gap.

**Single-source findings to verify**: Geographic error (Claude), privacy/governance (Codex), Generator requirements (Claude).

---

## Recommendations

### Must Fix (before implementation)
1. **K variable collision** — rename bridging knowledge to avoid confusion with STM size K
2. **K×d bits claim** — this conflates dimensions with information capacity
3. **Define success criteria** — quantitative go/no-go thresholds for each phase

### Should Address
- Add computational cost estimates for Phase 1a
- Address adversarial robustness / safety considerations
- Clarify the v1→v2 architectural bridge for streaming
- Strengthen the logic connecting "optimization succeeds" to "factored representations"

### Overall Assessment
The core hypothesis and architecture are clearly articulated. The document would benefit from more operational detail (compute, success criteria, failure handling) before implementation begins.

---

*Review generated via MMS (Multi-Model Synthesis) with Claude + Codex evaluation*
*Date: 2026-01-17*
