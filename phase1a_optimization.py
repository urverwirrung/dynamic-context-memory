"""
Phase 1a: STM Target Generation via Optimization

This module implements the core optimization loop for finding content embeddings
that minimize Generator loss on (Q, A) pairs. Each successful optimization
produces a training target for the Diffuser.

Reference: dcm-plan.md, Part III > Target Generation via Optimization
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for content embedding optimization."""
    K: int                          # Number of content embedding slots
    d: int                          # Embedding dimension (must match Generator)
    num_restarts: int = 5           # Independent optimization runs per (Q, A)
    num_steps: int = 800            # Gradient steps per run
    lr: float = 0.001               # Initial learning rate (reduced from 0.01 to avoid NaN)
    init_scale: float = 0.02        # Scale for random initialization
    use_scheduler: bool = True      # Use cosine annealing LR schedule
    scheduler_t_max: int = 800      # Fixed T_max for cosine annealing (decoupled from num_steps)
    use_warm_restarts: bool = False # Use CosineAnnealingWarmRestarts instead of CosineAnnealingLR
    warm_restart_t0: int = 400      # Initial restart period (T_0)
    warm_restart_t_mult: int = 2    # Multiplier for restart period (T_mult): 400, 800, 1600...
    use_reduce_on_plateau: bool = False  # Additionally reduce LR when loss plateaus (disabled: conflicts with scheduled sampling)
    plateau_patience: int = 100     # Steps without improvement before reducing LR
    plateau_factor: float = 0.5     # Factor to reduce LR by
    min_lr: float = 1e-6            # Minimum learning rate
    grad_clip: Optional[float] = 1.0  # Gradient clipping (None to disable)
    # Adaptive reconstruction checking - frequency inversely proportional to loss delta
    recon_check_min_interval: int = 250   # Minimum steps between checks (when loss plateaus)
    recon_check_max_interval: int = 1000  # Maximum steps between checks (when loss dropping fast)
    recon_check_window: int = 50          # Window size for computing loss delta
    recon_check_delta_threshold: float = 0.001  # Delta below this = plateau, check frequently (lower = slower convergence to min)
    reconstruction_prefix_len: int = 0    # 0 = require full match; >0 = require first N tokens
    # EOS-based convergence: check P(EOS) at stopping position instead of exact match
    use_eos_convergence: bool = True      # Use P(EOS) threshold for convergence (softer than exact match)
    eos_prob_threshold: float = 0.5       # P(EOS) must be above this to consider converged
    # Inference-time anti-repetition (for reconstruction checking)
    repetition_penalty: float = 1.0       # >1.0 discourages repetition. Typical: 1.1-1.3
    no_repeat_ngram_size: int = 0         # If >0, prevent any n-gram from appearing twice
    # Scheduled sampling (Priority 1): gradually expose model to its own predictions
    use_scheduled_sampling: bool = False  # Enable scheduled sampling during training
    ss_initial_epsilon: float = 1.0       # Start with pure teacher forcing
    ss_final_epsilon: float = 0.2         # End with mostly model predictions
    ss_decay_type: str = "inverse_sigmoid"  # "linear", "exponential", or "inverse_sigmoid"
    ss_decay_k: float = 5.0               # Parameter for inverse sigmoid (higher = slower decay, 5-10 recommended)
    ss_plateau_nudge: bool = True         # Nudge epsilon down when loss plateaus (escape local minima)
    ss_nudge_patience: int = 75           # Steps without improvement before nudging epsilon
    ss_nudge_factor: float = 0.75         # Multiply epsilon by this factor when nudging DOWN
    ss_nudge_min_epsilon: float = 0.1     # Floor to maintain some teacher forcing (was 0.0, caused collapse)
    ss_nudge_lr_boost: float = 2.0        # Multiply LR by this factor when epsilon is nudged (helps adapt to new regime)
    # Bidirectional epsilon adjustment (escape degenerate local minima)
    ss_nudge_up_on_degenerate: bool = True  # Increase epsilon if reconstruction degenerates
    ss_nudge_up_factor: float = 1.5       # Multiply epsilon by this when degenerating (nudge up)
    ss_nudge_up_max_epsilon: float = 0.8  # Max epsilon when nudging up (don't go back to pure TF)
    ss_degenerate_repeat_threshold: int = 3  # Consecutive same-token count to detect degeneration
    # Semantic loss: reward semantic similarity instead of exact token match
    use_semantic_loss: bool = False         # Use embedding similarity loss instead of/alongside CE
    semantic_loss_weight: float = 1.0       # Weight for semantic loss vs CE (0 = pure CE, 1 = pure semantic)
    semantic_loss_temperature: float = 1.0  # Softmax temperature for semantic loss (lower = sharper)
    semantic_sentence_weight: float = 0.5   # Weight for sentence-level vs token-level (0 = token only, 1 = sentence only)
    eos_penalty_weight: float = 1.0         # Weight for extra token penalty (penalize non-EOS after target)
    # Unlikelihood loss (Priority 2): penalize repetition during training
    use_unlikelihood: bool = False        # Enable unlikelihood training
    unlikelihood_alpha: float = 0.5       # Weight for unlikelihood loss (0.5-1.0 recommended)


@dataclass
class OptimizationResult:
    """Result of optimizing content embeddings for a (Q, A) pair."""
    content: torch.Tensor           # Shape (K, d) - the optimized embeddings
    loss: float                     # Final combined loss
    loss_q: float                   # Final Q reconstruction loss
    loss_a: float                   # Final A reconstruction loss
    converged: bool                 # Whether loss < threshold
    num_restarts_tried: int         # How many restarts were attempted
    loss_history: List[float]       # Loss curve from best run
    steps_to_converge: int = 0      # Step at which convergence was achieved (0 = never)


class ContentOptimizer:
    """
    Optimizes content embeddings for (Q, A) pairs against a frozen Generator.

    The optimization objective:
        content* = argmin_c [ L_gen(Q | Q_mode, c) + L_gen(A | A_mode, c) ]

    Where Q_mode and A_mode are fixed embeddings that route the Generator
    to produce either the question or answer from the shared content.
    """

    def __init__(
        self,
        generator: nn.Module,
        generator_loss_fn: Callable,
        Q_mode: torch.Tensor,
        A_mode: torch.Tensor,
        config: OptimizationConfig,
        device: torch.device = None,
        eos_token_id: Optional[int] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Args:
            generator: Frozen language model
            generator_loss_fn: Function(model, target_tokens, prompt_embeds) -> loss
            Q_mode: Fixed mode embedding for question generation, shape (1, d)
            A_mode: Fixed mode embedding for answer generation, shape (1, d)
            config: Optimization hyperparameters
            device: Torch device
            eos_token_id: EOS token ID for convergence checking
            tokenizer: Optional tokenizer for decoding during logging
        """
        self.generator = generator
        self.generator_loss_fn = generator_loss_fn
        self.Q_mode = Q_mode
        self.A_mode = A_mode
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer

        # Ensure generator is frozen
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    def _initialize_content(self) -> torch.Tensor:
        """Initialize content embeddings with small random values."""
        # Use float32 for optimization stability (converted to model dtype in forward pass)
        content = torch.randn(
            self.config.K,
            self.config.d,
            device=self.device,
            dtype=torch.float32,
        ) * self.config.init_scale
        content.requires_grad_(True)
        return content

    def _compute_loss(
        self,
        content: torch.Tensor,
        Q_tokens: torch.Tensor,
        A_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss for Q and A reconstruction.

        Supports semantic loss when config.use_semantic_loss is True.

        Returns:
            total_loss, loss_q, loss_a
        """
        # Construct prompt embeddings: [mode, content...]
        prompt_q = torch.cat([self.Q_mode, content], dim=0)  # (1+K, d)
        prompt_a = torch.cat([self.A_mode, content], dim=0)  # (1+K, d)

        if self.config.use_semantic_loss:
            # Semantic similarity loss (partial credit for similar tokens)
            sem_loss_q, _, _ = hf_semantic_loss_fn(
                self.generator,
                Q_tokens,
                prompt_q,
                temperature=self.config.semantic_loss_temperature,
                sentence_weight=self.config.semantic_sentence_weight,
                eos_token_id=self.eos_token_id,
                eos_penalty_weight=self.config.eos_penalty_weight,
            )
            sem_loss_a, _, _ = hf_semantic_loss_fn(
                self.generator,
                A_tokens,
                prompt_a,
                temperature=self.config.semantic_loss_temperature,
                sentence_weight=self.config.semantic_sentence_weight,
                eos_token_id=self.eos_token_id,
                eos_penalty_weight=self.config.eos_penalty_weight,
            )

            # Blend with CE loss based on weight
            # weight=1.0 means pure semantic, weight=0.0 means pure CE
            w = self.config.semantic_loss_weight
            if w < 1.0:
                ce_loss_q = self.generator_loss_fn(self.generator, Q_tokens, prompt_q)
                ce_loss_a = self.generator_loss_fn(self.generator, A_tokens, prompt_a)
                loss_q = w * sem_loss_q + (1 - w) * ce_loss_q
                loss_a = w * sem_loss_a + (1 - w) * ce_loss_a
            else:
                loss_q = sem_loss_q
                loss_a = sem_loss_a
        else:
            # Standard cross-entropy loss
            loss_q = self.generator_loss_fn(self.generator, Q_tokens, prompt_q)
            loss_a = self.generator_loss_fn(self.generator, A_tokens, prompt_a)

        return loss_q + loss_a, loss_q, loss_a

    def _compute_check_interval(self, loss_history: List[float]) -> int:
        """
        Compute adaptive check interval based on loss delta.

        When loss is dropping fast (large delta), check rarely.
        When loss plateaus (small delta), check frequently.
        """
        window = self.config.recon_check_window
        min_interval = self.config.recon_check_min_interval
        max_interval = self.config.recon_check_max_interval
        delta_threshold = self.config.recon_check_delta_threshold

        if len(loss_history) < window:
            return max_interval  # Not enough history, check rarely

        # Compute delta over window
        recent = loss_history[-window:]
        delta = recent[0] - recent[-1]  # Positive if loss is decreasing

        if delta <= 0:
            # Loss increasing or flat - check frequently
            return min_interval

        if delta < delta_threshold:
            # Loss plateauing - check frequently
            return min_interval

        # Scale interval based on delta: larger delta -> larger interval
        ratio = delta / (delta + delta_threshold)
        interval = int(min_interval + (max_interval - min_interval) * ratio)
        return min(max_interval, max(min_interval, interval))

    def _check_convergence(
        self,
        content: torch.Tensor,
        Q_tokens: torch.Tensor,
        A_tokens: torch.Tensor,
        Q_list: List[int],
        A_list: List[int],
    ) -> Tuple[bool, bool, List[int], List[int]]:
        """
        Check if both Q and A reconstruct correctly.

        Returns:
            (q_converged, a_converged, q_generated, a_generated)
        """
        prefix_len = self.config.reconstruction_prefix_len
        q_len = len(Q_list)
        a_len = len(A_list)

        # Decode Q with Q_mode prefix
        prompt_q = torch.cat([self.Q_mode, content.detach()], dim=0)
        q_generated = greedy_decode_tokens(
            self.generator,
            prompt_q,
            max_new_tokens=q_len + 5,
            eos_token_id=self.eos_token_id,
            repetition_penalty=self.config.repetition_penalty,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
        )

        # Decode A with A_mode prefix
        prompt_a = torch.cat([self.A_mode, content.detach()], dim=0)
        a_generated = greedy_decode_tokens(
            self.generator,
            prompt_a,
            max_new_tokens=a_len + 5,
            eos_token_id=self.eos_token_id,
            repetition_penalty=self.config.repetition_penalty,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
        )

        # Check exact match for Q
        q_exact = check_reconstruction(q_generated, Q_list, prefix_len)
        q_converged = q_exact

        # Check EOS convergence for Q if exact match failed
        if not q_exact and self.config.use_eos_convergence and self.eos_token_id is not None:
            q_content = [t for t in Q_list if t != self.eos_token_id]
            q_content_len = len(q_content)
            q_content_matches = (
                len(q_generated) >= q_content_len and
                q_generated[:q_content_len] == q_content
            )
            if q_content_matches:
                q_eos_prob = get_eos_probability(
                    self.generator, Q_tokens, prompt_q, self.eos_token_id
                )
                q_converged = q_eos_prob >= self.config.eos_prob_threshold

        # Check exact match for A
        a_exact = check_reconstruction(a_generated, A_list, prefix_len)
        a_converged = a_exact

        # Check EOS convergence for A if exact match failed
        if not a_exact and self.config.use_eos_convergence and self.eos_token_id is not None:
            a_content = [t for t in A_list if t != self.eos_token_id]
            a_content_len = len(a_content)
            a_content_matches = (
                len(a_generated) >= a_content_len and
                a_generated[:a_content_len] == a_content
            )
            if a_content_matches:
                a_eos_prob = get_eos_probability(
                    self.generator, A_tokens, prompt_a, self.eos_token_id
                )
                a_converged = a_eos_prob >= self.config.eos_prob_threshold

        return q_converged, a_converged, q_generated, a_generated

    def _single_run(
        self,
        Q_tokens: torch.Tensor,
        A_tokens: torch.Tensor,
        Q_list: List[int],
        A_list: List[int],
        pbar: Optional["tqdm.tqdm"] = None,
        example_idx: int = 0,
        num_examples: int = 1,
        restart_idx: int = 0,
    ) -> Tuple[torch.Tensor, float, float, float, List[float], int, bool, List[int], List[int]]:
        """
        Single optimization run from random initialization.

        Returns:
            (best_content, best_loss, best_loss_q, best_loss_a, loss_history,
             steps_to_converge, converged, q_generated, a_generated)

        converged = True means BOTH Q and A reconstruct correctly
        """
        content = self._initialize_content()
        optimizer = torch.optim.Adam([content], lr=self.config.lr)

        scheduler = None
        if self.config.use_scheduler:
            if self.config.use_warm_restarts:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.config.warm_restart_t0,
                    T_mult=self.config.warm_restart_t_mult,
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.config.scheduler_t_max
                )

        loss_history = []
        best_content = content.detach().clone()
        best_loss = float('inf')
        best_loss_q = float('inf')
        best_loss_a = float('inf')
        converged = False
        steps_to_converge = 0
        last_q_generated = []
        last_a_generated = []
        steps_since_last_check = 0
        current_lr = self.config.lr

        for step in range(self.config.num_steps):
            optimizer.zero_grad()

            loss, loss_q, loss_a = self._compute_loss(content, Q_tokens, A_tokens)

            loss.backward()

            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([content], self.config.grad_clip)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

            loss_val = loss.item()
            loss_q_val = loss_q.item()
            loss_a_val = loss_a.item()

            # Detect NaN/Inf and abort this run early
            if math.isnan(loss_val) or math.isinf(loss_val):
                if pbar is not None:
                    pbar.update(self.config.num_steps - step - 1)
                return (best_content, best_loss, best_loss_q, best_loss_a, loss_history,
                        steps_to_converge, converged, last_q_generated, last_a_generated)

            loss_history.append(loss_val)
            steps_since_last_check += 1

            if loss_val < best_loss:
                best_loss = loss_val
                best_loss_q = loss_q_val
                best_loss_a = loss_a_val
                best_content = content.detach().clone()

            # Adaptive reconstruction checking
            check_interval = self._compute_check_interval(loss_history)
            should_check = (
                steps_since_last_check >= check_interval or
                step == self.config.num_steps - 1
            )

            if should_check:
                steps_since_last_check = 0
                q_conv, a_conv, q_gen, a_gen = self._check_convergence(
                    content, Q_tokens, A_tokens, Q_list, A_list
                )
                last_q_generated = q_gen
                last_a_generated = a_gen

                # Both must converge for joint optimization to succeed
                if q_conv and a_conv:
                    converged = True
                    steps_to_converge = step + 1
                    best_content = content.detach().clone()
                    best_loss = loss_val
                    best_loss_q = loss_q_val
                    best_loss_a = loss_a_val

                    if pbar is not None:
                        pbar.set_postfix(loss=f"{loss_val:.6f}", status="✓ BOTH")
                        pbar.update(self.config.num_steps - step - 1)
                    break

            if pbar is not None:
                pbar.set_description(
                    f"ex {example_idx+1}/{num_examples} | restart {restart_idx+1}/{self.config.num_restarts}"
                )
                pbar.set_postfix(loss=f"{loss_val:.6f}", lr=f"{current_lr:.1e}")
                pbar.update(1)

        # Final reconstruction check if we didn't converge during training
        if not converged:
            q_conv, a_conv, last_q_generated, last_a_generated = self._check_convergence(
                best_content, Q_tokens, A_tokens, Q_list, A_list
            )
            if q_conv and a_conv:
                converged = True
                steps_to_converge = self.config.num_steps

        return (best_content, best_loss, best_loss_q, best_loss_a, loss_history,
                steps_to_converge, converged, last_q_generated, last_a_generated)

    def optimize(
        self,
        Q_tokens: torch.Tensor,
        A_tokens: torch.Tensor,
        pbar: Optional["tqdm.tqdm"] = None,
        example_idx: int = 0,
        num_examples: int = 1,
    ) -> OptimizationResult:
        """
        Find optimal content embeddings for a (Q, A) pair.

        Uses multiple restarts and returns the best result.
        Convergence = BOTH Q and A reconstruct correctly.

        Args:
            Q_tokens: Tokenized question, shape (seq_len_q,)
            A_tokens: Tokenized answer, shape (seq_len_a,)
            pbar: Optional tqdm progress bar for step-level updates
            example_idx: Current example index (for progress display)
            num_examples: Total number of examples (for progress display)

        Returns:
            OptimizationResult with best content embeddings and metadata
        """
        # Convert to lists for reconstruction checking
        Q_list = Q_tokens.tolist()
        A_list = A_tokens.tolist()

        best_content = None
        best_loss = float('inf')
        best_loss_q = float('inf')
        best_loss_a = float('inf')
        best_history = []
        best_converged = False
        best_steps_to_converge = 0
        restarts_tried = 0

        for restart in range(self.config.num_restarts):
            restarts_tried = restart + 1
            (content, loss, loss_q, loss_a, history, steps_to_converge,
             converged, q_gen, a_gen) = self._single_run(
                Q_tokens, A_tokens, Q_list, A_list,
                pbar=pbar, example_idx=example_idx, num_examples=num_examples,
                restart_idx=restart,
            )

            # If this run converged (both Q and A reconstruct), we're done
            if converged:
                return OptimizationResult(
                    content=content,
                    loss=loss,
                    loss_q=loss_q,
                    loss_a=loss_a,
                    converged=True,
                    num_restarts_tried=restarts_tried,
                    loss_history=history,
                    steps_to_converge=steps_to_converge,
                )

            # Otherwise track best loss (even though it didn't converge)
            if loss < best_loss:
                best_loss = loss
                best_loss_q = loss_q
                best_loss_a = loss_a
                best_content = content
                best_history = history

        # No restart achieved convergence - return best attempt
        return OptimizationResult(
            content=best_content,
            loss=best_loss,
            loss_q=best_loss_q,
            loss_a=best_loss_a,
            converged=False,
            num_restarts_tried=restarts_tried,
            loss_history=best_history,
            steps_to_converge=0,
        )


# -----------------------------------------------------------------------------
# Single-target optimization (diagnostic)
# -----------------------------------------------------------------------------

@dataclass
class SingleTargetResult:
    """Result of single-target optimization diagnostic."""
    content: torch.Tensor           # Shape (K, d) - the optimized embeddings
    loss: float                     # Final loss
    converged: bool                 # Whether reconstruction succeeded
    num_restarts_tried: int
    loss_history: List[float]
    steps_to_converge: int = 0      # Step at which reconstruction succeeded (0 = never)
    generated_tokens: Optional[List[int]] = None  # What model actually generates
    target_tokens: Optional[List[int]] = None     # What we wanted


def greedy_decode_tokens(
    model: nn.Module,
    prompt_embeds: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> List[int]:
    """
    Greedy decode from soft prompt embeddings, returning token IDs.

    Args:
        model: HuggingFace model
        prompt_embeds: Shape (K, d) or (1, K, d)
        max_new_tokens: Maximum tokens to generate
        eos_token_id: Stop at this token (optional)
        repetition_penalty: Divide logits of already-generated tokens by this value.
            >1.0 discourages repetition, 1.0 = no penalty. Typical: 1.1-1.3
        no_repeat_ngram_size: If >0, prevent any n-gram from appearing twice.
            E.g., 2 = no repeated bigrams. 0 = disabled.

    Returns:
        List of generated token IDs
    """
    if prompt_embeds.dim() == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)  # (1, K, d)

    device = prompt_embeds.device
    model_dtype = next(model.parameters()).dtype
    current_embeds = prompt_embeds.to(dtype=model_dtype)

    generated_ids = []

    def _get_banned_ngram_tokens(ngram_size: int, prev_ids: List[int]) -> set:
        """Find tokens that would create a repeated n-gram."""
        if len(prev_ids) < ngram_size - 1:
            return set()

        banned = set()
        # Look for (ngram_size - 1) prefix matches in history
        current_prefix = tuple(prev_ids[-(ngram_size - 1):])
        for i in range(len(prev_ids) - ngram_size + 1):
            historical_prefix = tuple(prev_ids[i:i + ngram_size - 1])
            if historical_prefix == current_prefix:
                # The token that followed this prefix would create a repeat
                banned.add(prev_ids[i + ngram_size - 1])
        return banned

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(inputs_embeds=current_embeds)
            next_token_logits = outputs.logits[0, -1, :].clone()

            # Apply repetition penalty to already-generated tokens
            if repetition_penalty != 1.0 and generated_ids:
                for token_id in set(generated_ids):
                    # Penalty: divide positive logits, multiply negative logits
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= repetition_penalty
                    else:
                        next_token_logits[token_id] *= repetition_penalty

            # Apply n-gram blocking
            if no_repeat_ngram_size > 0 and generated_ids:
                banned_tokens = _get_banned_ngram_tokens(no_repeat_ngram_size, generated_ids)
                for token_id in banned_tokens:
                    next_token_logits[token_id] = float('-inf')

            next_token_id = torch.argmax(next_token_logits).item()
            generated_ids.append(next_token_id)

            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            next_embed = model.get_input_embeddings()(
                torch.tensor([[next_token_id]], device=device)
            )
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)

    return generated_ids


def check_reconstruction(
    generated: List[int],
    target: List[int],
    prefix_len: int = 0,
) -> bool:
    """
    Check if generated tokens match target.

    Args:
        generated: Generated token IDs
        target: Target token IDs
        prefix_len: If > 0, only check first N tokens. If 0, require full match.

    Returns:
        True if reconstruction succeeded
    """
    if prefix_len > 0:
        return generated[:prefix_len] == target[:prefix_len]
    else:
        return generated[:len(target)] == target


def get_eos_probability(
    model: nn.Module,
    target_tokens: torch.Tensor,
    prompt_embeds: torch.Tensor,
    eos_token_id: int,
) -> float:
    """
    Get the probability the model assigns to EOS at the position after target content.

    If target = [tok1, tok2, ..., tokN, EOS], we check P(EOS | tok1...tokN).
    This measures how well the embedding teaches the model to stop.

    Args:
        model: HuggingFace model
        target_tokens: Target sequence INCLUDING EOS at the end
        prompt_embeds: Soft prompt embeddings
        eos_token_id: The EOS token ID

    Returns:
        Probability of EOS at the stopping position (0.0 to 1.0)
    """
    device = prompt_embeds.device
    model_dtype = next(model.parameters()).dtype
    embed_layer = model.get_input_embeddings()

    prompt_embeds_cast = prompt_embeds.to(dtype=model_dtype)

    # Get embeddings for target tokens (excluding the final EOS - we want to predict it)
    # If target is [A, B, C, EOS], we feed [prompt, A, B, C] and check P(EOS) at position C
    target_without_eos = target_tokens[:-1] if target_tokens[-1].item() == eos_token_id else target_tokens
    target_embeds = embed_layer(target_without_eos)

    # Forward pass
    full_embeds = torch.cat([prompt_embeds_cast, target_embeds], dim=0).unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs_embeds=full_embeds)
        # Logits at last position predict what comes next (should be EOS)
        last_logits = outputs.logits[0, -1, :]  # (vocab_size,)
        probs = torch.softmax(last_logits, dim=-1)
        eos_prob = probs[eos_token_id].item()

    return eos_prob


def detect_degenerate_output(
    generated: List[int],
    repeat_threshold: int = 3,
) -> bool:
    """
    Detect if output has degenerated into repetitive garbage.

    Signs of degeneration:
    - Same token repeated many times in a row
    - Very short repeating patterns (e.g., "AB AB AB")

    Args:
        generated: Generated token IDs
        repeat_threshold: Number of consecutive same tokens to trigger

    Returns:
        True if output appears degenerate
    """
    if len(generated) < repeat_threshold:
        return False

    # Check for consecutive same tokens
    consecutive = 1
    for i in range(1, len(generated)):
        if generated[i] == generated[i - 1]:
            consecutive += 1
            if consecutive >= repeat_threshold:
                return True
        else:
            consecutive = 1

    # Check for short repeating patterns (bigrams, trigrams)
    # e.g., [A, B, A, B, A, B] or [A, B, C, A, B, C]
    for pattern_len in [2, 3]:
        if len(generated) >= pattern_len * 3:  # Need at least 3 repetitions
            pattern = tuple(generated[:pattern_len])
            repeats = 0
            for i in range(0, len(generated) - pattern_len + 1, pattern_len):
                if tuple(generated[i:i + pattern_len]) == pattern:
                    repeats += 1
                else:
                    break
            if repeats >= 3:
                return True

    return False


class SingleTargetOptimizer:
    """
    Diagnostic: optimize content for a single target sequence.

    This is a simpler test than joint (Q, A) optimization. If single-target
    fails, the joint objective won't work either. Use this to validate that
    the Generator's embedding space is addressable before attempting the
    harder dual-mode optimization.

    Objective: c* = argmin_c [ L_gen(target | c) ]

    No mode embedding — just raw content → target reconstruction.
    Convergence = reconstruction success, not loss threshold.
    """

    def __init__(
        self,
        generator: nn.Module,
        generator_loss_fn: Callable,
        config: OptimizationConfig,
        device: torch.device = None,
        eos_token_id: Optional[int] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.generator = generator
        self.generator_loss_fn = generator_loss_fn
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer  # Optional, for decoding during nudge checks

        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    def _initialize_content(self) -> torch.Tensor:
        # Use float32 for optimization stability (converted to model dtype in forward pass)
        content = torch.randn(
            self.config.K,
            self.config.d,
            device=self.device,
            dtype=torch.float32,
        ) * self.config.init_scale
        content.requires_grad_(True)
        return content

    def _compute_check_interval(self, loss_history: List[float]) -> int:
        """
        Compute adaptive check interval based on loss delta.

        When loss is dropping fast (large delta), check rarely.
        When loss plateaus (small delta), check frequently.
        """
        window = self.config.recon_check_window
        min_interval = self.config.recon_check_min_interval
        max_interval = self.config.recon_check_max_interval
        delta_threshold = self.config.recon_check_delta_threshold

        if len(loss_history) < window:
            return max_interval  # Not enough history, check rarely

        # Compute delta over window
        recent = loss_history[-window:]
        delta = recent[0] - recent[-1]  # Positive if loss is decreasing

        if delta <= 0:
            # Loss increasing or flat - check frequently
            return min_interval

        if delta < delta_threshold:
            # Loss plateauing - check frequently
            return min_interval

        # Scale interval based on delta: larger delta -> larger interval
        # interval = min + (max - min) * (delta / (delta + threshold))
        ratio = delta / (delta + delta_threshold)
        interval = int(min_interval + (max_interval - min_interval) * ratio)
        return min(max_interval, max(min_interval, interval))

    def _single_run(
        self,
        target_tokens: torch.Tensor,
        target_list: List[int],
        pbar: Optional["tqdm.tqdm"] = None,
        example_idx: int = 0,
        num_examples: int = 1,
        restart_idx: int = 0,
    ) -> Tuple[torch.Tensor, float, List[float], int, bool, List[int]]:
        """
        Returns: (best_content, best_loss, loss_history, steps_to_converge, converged, generated_tokens)

        converged = True means reconstruction succeeded (not just low loss)
        steps_to_converge = step at which reconstruction succeeded (0 if never)

        Uses adaptive reconstruction checking: checks frequently when loss plateaus,
        rarely when loss is still dropping fast.
        """
        content = self._initialize_content()
        optimizer = torch.optim.Adam([content], lr=self.config.lr)

        scheduler = None
        if self.config.use_scheduler:
            if self.config.use_warm_restarts:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=self.config.warm_restart_t0,
                    T_mult=self.config.warm_restart_t_mult,
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.config.scheduler_t_max
                )

        loss_history = []
        best_content = content.detach().clone()
        best_loss = float('inf')
        converged = False
        steps_to_converge = 0
        last_generated = []
        steps_since_last_check = 0

        # Plateau-based LR reduction tracking
        steps_since_improvement = 0
        plateau_best_loss = float('inf')
        current_lr = self.config.lr

        target_len = len(target_list)
        prefix_len = self.config.reconstruction_prefix_len

        # Check if we're using scheduled sampling or unlikelihood
        use_advanced_loss = (
            self.config.use_scheduled_sampling or
            self.config.use_unlikelihood
        )
        use_semantic = self.config.use_semantic_loss

        # Plateau-triggered epsilon nudge tracking
        epsilon_multiplier = 1.0  # Accumulated nudges (gets multiplied down)
        nudge_steps_since_improvement = 0
        nudge_best_loss = float('inf')

        for step in range(self.config.num_steps):
            optimizer.zero_grad()

            if use_semantic:
                # Semantic similarity loss (partial credit for similar tokens)
                sem_loss, token_loss, sentence_loss = hf_semantic_loss_fn(
                    self.generator,
                    target_tokens,
                    content,
                    temperature=self.config.semantic_loss_temperature,
                    sentence_weight=self.config.semantic_sentence_weight,
                    eos_token_id=self.eos_token_id,
                    eos_penalty_weight=self.config.eos_penalty_weight,
                )
                # Blend with CE loss based on weight
                # weight=1.0 means pure semantic, weight=0.0 means pure CE
                w = self.config.semantic_loss_weight
                if w < 1.0:
                    ce_loss = self.generator_loss_fn(self.generator, target_tokens, content)
                    loss = w * sem_loss + (1 - w) * ce_loss
                else:
                    loss = sem_loss
                epsilon = 1.0  # No scheduled sampling with semantic loss for now
            elif use_advanced_loss:
                # Compute epsilon for scheduled sampling
                base_epsilon = compute_epsilon(step, self.config.num_steps, self.config)

                # Apply plateau nudge if enabled
                if self.config.ss_plateau_nudge:
                    epsilon = max(
                        self.config.ss_nudge_min_epsilon,
                        base_epsilon * epsilon_multiplier
                    )
                else:
                    epsilon = base_epsilon

                ul_alpha = self.config.unlikelihood_alpha if self.config.use_unlikelihood else 0.0

                # Use fast scheduled sampling loss
                loss, ce_loss, ul_loss = hf_scheduled_sampling_loss_fn_fast(
                    self.generator,
                    target_tokens,
                    content,
                    epsilon=epsilon,
                    unlikelihood_alpha=ul_alpha,
                )
            else:
                loss = self.generator_loss_fn(self.generator, target_tokens, content)
                epsilon = 1.0  # For display purposes

            loss.backward()

            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([content], self.config.grad_clip)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

            loss_val = loss.item()

            # Detect NaN/Inf and abort this run early
            if math.isnan(loss_val) or math.isinf(loss_val):
                if pbar is not None:
                    pbar.update(self.config.num_steps - step - 1)
                return best_content, best_loss, loss_history, steps_to_converge, converged, last_generated

            loss_history.append(loss_val)
            steps_since_last_check += 1

            if loss_val < best_loss:
                best_loss = loss_val
                best_content = content.detach().clone()

            # Plateau-based LR reduction
            if self.config.use_reduce_on_plateau:
                if loss_val < plateau_best_loss - 1e-7:  # Meaningful improvement
                    plateau_best_loss = loss_val
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1

                if steps_since_improvement >= self.config.plateau_patience:
                    new_lr = max(current_lr * self.config.plateau_factor, self.config.min_lr)
                    if new_lr < current_lr:
                        current_lr = new_lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                        steps_since_improvement = 0
                        plateau_best_loss = loss_val  # Reset baseline

            # Plateau-triggered epsilon nudge (bidirectional: can nudge down OR up)
            # Only applies to scheduled sampling, not semantic loss
            if use_advanced_loss and not use_semantic and self.config.ss_plateau_nudge:
                if loss_val < nudge_best_loss - 1e-6:  # Meaningful improvement
                    nudge_best_loss = loss_val
                    nudge_steps_since_improvement = 0
                else:
                    nudge_steps_since_improvement += 1

                if nudge_steps_since_improvement >= self.config.ss_nudge_patience:
                    # Check current reconstruction quality before deciding nudge direction
                    nudge_generated = greedy_decode_tokens(
                        self.generator,
                        content.detach(),
                        max_new_tokens=target_len + 5,
                        eos_token_id=self.eos_token_id,
                        repetition_penalty=self.config.repetition_penalty,
                        no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                    )
                    nudge_converged = check_reconstruction(nudge_generated, target_list, prefix_len)
                    is_degenerate = detect_degenerate_output(
                        nudge_generated,
                        repeat_threshold=self.config.ss_degenerate_repeat_threshold,
                    )

                    old_multiplier = epsilon_multiplier
                    old_lr = current_lr
                    nudge_direction = None

                    if is_degenerate and self.config.ss_nudge_up_on_degenerate:
                        # Output is degenerate - nudge epsilon UP to get more teacher forcing
                        max_mult = self.config.ss_nudge_up_max_epsilon / max(base_epsilon, 0.01)
                        epsilon_multiplier = min(max_mult, epsilon_multiplier * self.config.ss_nudge_up_factor)
                        nudge_direction = "UP"
                    else:
                        # Not degenerate - nudge epsilon DOWN to inject more variance
                        min_mult = self.config.ss_nudge_min_epsilon / max(base_epsilon, 0.01)
                        epsilon_multiplier = max(min_mult, epsilon_multiplier * self.config.ss_nudge_factor)
                        nudge_direction = "DOWN"

                    # Only act if multiplier actually changed
                    if epsilon_multiplier != old_multiplier:
                        # Boost LR to help adapt to new epsilon regime
                        new_lr = min(current_lr * self.config.ss_nudge_lr_boost, self.config.lr)
                        if new_lr > current_lr:
                            current_lr = new_lr
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = current_lr

                        status = "✓" if nudge_converged else ("⚠ DEGEN" if is_degenerate else "✗")
                        direction_symbol = "↑" if nudge_direction == "UP" else "↓"

                        if pbar is not None:
                            from tqdm import tqdm
                            tqdm.write(f"    {direction_symbol} Epsilon {nudge_direction} @ loss={loss_val:.6f}: ε {base_epsilon:.5f}×{old_multiplier:.5f}→{epsilon_multiplier:.5f}, LR: {old_lr:.1e}→{current_lr:.1e}")
                            # Show reconstruction comparison
                            if self.tokenizer is not None:
                                target_text = self.tokenizer.decode(target_list, skip_special_tokens=True)
                                generated_text = self.tokenizer.decode(nudge_generated, skip_special_tokens=True)
                                tqdm.write(f"    {status} Target:    {target_text}")
                                tqdm.write(f"    {status} Generated: {generated_text}")
                            else:
                                tqdm.write(f"    {status} Target tokens:    {target_list}")
                                tqdm.write(f"    {status} Generated tokens: {nudge_generated}")

                        # Update convergence if we hit it
                        if nudge_converged and not converged:
                            converged = True
                            steps_to_converge = step + 1
                            best_content = content.detach().clone()
                            best_loss = loss_val
                        last_generated = nudge_generated

                    nudge_steps_since_improvement = 0
                    nudge_best_loss = loss_val  # Reset baseline

            # Adaptive reconstruction checking
            check_interval = self._compute_check_interval(loss_history)
            should_check = (
                steps_since_last_check >= check_interval or
                step == self.config.num_steps - 1
            )

            if should_check:
                steps_since_last_check = 0
                generated = greedy_decode_tokens(
                    self.generator,
                    content.detach(),
                    max_new_tokens=target_len + 5,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=self.config.repetition_penalty,
                    no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                )
                last_generated = generated

                # Check convergence: exact match OR (content match + high P(EOS))
                exact_match = check_reconstruction(generated, target_list, prefix_len)
                eos_converged = False
                eos_prob = None

                if not exact_match and self.config.use_eos_convergence and self.eos_token_id is not None:
                    # EOS convergence: content must be correct, just didn't stop cleanly
                    # Check if generated tokens match target content (excluding EOS)
                    target_content = [t for t in target_list if t != self.eos_token_id]
                    content_len = len(target_content)

                    # Generated must have at least the content tokens and they must match
                    content_matches = (
                        len(generated) >= content_len and
                        generated[:content_len] == target_content
                    )

                    if content_matches:
                        # Content is correct - check if P(EOS) at stopping position is high
                        eos_prob = get_eos_probability(
                            self.generator,
                            target_tokens,
                            content.detach(),
                            self.eos_token_id,
                        )
                        eos_converged = eos_prob >= self.config.eos_prob_threshold

                if exact_match or eos_converged:
                    converged = True
                    steps_to_converge = step + 1
                    best_content = content.detach().clone()
                    best_loss = loss_val

                    if pbar is not None:
                        if exact_match:
                            status = "✓ EXACT"
                        else:
                            status = f"✓ EOS p={eos_prob:.2f}"
                        pbar.set_postfix(loss=f"{loss_val:.8f}", status=status)
                        pbar.update(self.config.num_steps - step - 1)
                    break

            if pbar is not None:
                pbar.set_description(
                    f"ex {example_idx+1}/{num_examples} | restart {restart_idx+1}/{self.config.num_restarts}"
                )
                postfix = {"loss": f"{loss_val:.6f}", "lr": f"{current_lr:.1e}"}
                if use_advanced_loss:
                    postfix["ε"] = f"{epsilon:.5f}"
                pbar.set_postfix(**postfix)
                pbar.update(1)

        # Final reconstruction check if we didn't converge during training
        if not converged:
            last_generated = greedy_decode_tokens(
                self.generator,
                best_content,
                max_new_tokens=target_len + 5,
                eos_token_id=self.eos_token_id,
                repetition_penalty=self.config.repetition_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            )

        return best_content, best_loss, loss_history, steps_to_converge, converged, last_generated

    def _run_with_restarts(
        self,
        target_tokens: torch.Tensor,
        target_list: List[int],
        pbar: Optional["tqdm.tqdm"] = None,
        example_idx: int = 0,
        num_examples: int = 1,
    ) -> SingleTargetResult:
        """Run optimization with multiple restarts, updating shared progress bar.

        Convergence = reconstruction success, not loss threshold.
        Stops early if any restart achieves successful reconstruction.
        """
        best_content = None
        best_loss = float('inf')
        best_history = []
        best_steps_to_converge = 0
        best_converged = False
        best_generated = []
        restarts_tried = 0

        for restart in range(self.config.num_restarts):
            restarts_tried = restart + 1
            content, loss, history, steps_to_converge, converged, generated = self._single_run(
                target_tokens,
                target_list,
                pbar=pbar,
                example_idx=example_idx,
                num_examples=num_examples,
                restart_idx=restart,
            )

            # If this run converged (reconstruction success), we're done
            if converged:
                return SingleTargetResult(
                    content=content,
                    loss=loss,
                    converged=True,
                    num_restarts_tried=restarts_tried,
                    loss_history=history,
                    steps_to_converge=steps_to_converge,
                    generated_tokens=generated,
                    target_tokens=target_list,
                )

            # Otherwise track best loss (even though it didn't converge)
            if loss < best_loss:
                best_loss = loss
                best_content = content
                best_history = history
                best_generated = generated

        # No restart achieved reconstruction
        return SingleTargetResult(
            content=best_content,
            loss=best_loss,
            converged=False,
            num_restarts_tried=restarts_tried,
            loss_history=best_history,
            steps_to_converge=0,
            generated_tokens=best_generated,
            target_tokens=target_list,
        )

    def optimize(self, target_tokens: torch.Tensor, verbose: bool = False) -> SingleTargetResult:
        """
        Find content embeddings that reconstruct a single target sequence.

        Args:
            target_tokens: Tokenized target, shape (seq_len,)
            verbose: If True, show progress bar

        Returns:
            SingleTargetResult with optimized embeddings and diagnostics
        """
        from tqdm import tqdm

        target_list = target_tokens.tolist()

        if verbose:
            total_steps = self.config.num_steps * self.config.num_restarts
            with tqdm(total=total_steps, desc="[1/1] restart 1/1", dynamic_ncols=True) as pbar:
                return self._run_with_restarts(target_tokens, target_list, pbar=pbar, example_idx=0, num_examples=1)
        else:
            return self._run_with_restarts(target_tokens, target_list, pbar=None, example_idx=0, num_examples=1)

    def run_diagnostic(
        self,
        targets: List[torch.Tensor],
        names: Optional[List[str]] = None,
    ) -> dict:
        """
        Run single-target optimization on multiple sequences and report statistics.

        Use this to assess feasibility before joint optimization.
        Convergence = reconstruction success, not loss threshold.

        Args:
            targets: List of tokenized sequences
            names: Optional names for reporting

        Returns:
            Dict with convergence stats and per-target results
        """
        from tqdm import tqdm

        results = []
        converged_count = 0

        # Single progress bar over all steps across all examples and restarts
        total_steps = len(targets) * self.config.num_restarts * self.config.num_steps
        with tqdm(total=total_steps, desc="optimizing", dynamic_ncols=True) as pbar:
            for i, target in enumerate(targets):
                name = names[i] if names else f"target_{i}"
                target_list = target.tolist()

                # Debug: warn about suspiciously short targets
                if target.numel() <= 2:
                    tqdm.write(f"  ⚠ {name}: target has only {target.numel()} token(s) - may cause trivial loss")

                result = self._run_with_restarts(
                    target,
                    target_list,
                    pbar=pbar,
                    example_idx=i,
                    num_examples=len(targets),
                )
                results.append(result)

                if result.converged:
                    converged_count += 1

                status = "✓" if result.converged else "✗"
                steps_info = f"steps: {result.steps_to_converge}" if result.converged else "no convergence"
                tqdm.write(f"  {status} {name}: best_loss={result.loss:.8f} ({steps_info}, restarts: {result.num_restarts_tried})")

        losses = [r.loss for r in results]

        return {
            "converged": converged_count,
            "total": len(targets),
            "convergence_rate": converged_count / len(targets),
            "mean_loss": sum(losses) / len(losses),
            "min_loss": min(losses),
            "max_loss": max(losses),
            "results": results,
        }


def create_mode_embeddings(
    generator: nn.Module,
    q_token_id: int,
    a_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create fixed mode embeddings from arbitrary token IDs.

    The mode embeddings are frozen and never optimized. They serve purely
    to route the Generator toward producing Q or A from the shared content.

    Args:
        generator: Model with embedding layer
        q_token_id: Token ID to use for Q mode (e.g., token for "Q")
        a_token_id: Token ID to use for A mode (e.g., token for "A")

    Returns:
        Q_mode, A_mode tensors of shape (1, d)
    """
    # Get embedding layer (adjust attribute path for your model)
    if hasattr(generator, 'get_input_embeddings'):
        embed = generator.get_input_embeddings()
    elif hasattr(generator, 'embed_tokens'):
        embed = generator.embed_tokens
    elif hasattr(generator, 'transformer') and hasattr(generator.transformer, 'wte'):
        embed = generator.transformer.wte
    else:
        raise ValueError("Cannot find embedding layer in generator")

    # Get device from embedding weights
    device = embed.weight.device

    with torch.no_grad():
        Q_mode = embed(torch.tensor([q_token_id], device=device)).detach()  # (1, d)
        A_mode = embed(torch.tensor([a_token_id], device=device)).detach()  # (1, d)

    return Q_mode, A_mode


# -----------------------------------------------------------------------------
# HuggingFace generator loss function
# -----------------------------------------------------------------------------

def hf_generator_loss_fn(
    model: nn.Module,
    target_tokens: torch.Tensor,
    prompt_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for generating target_tokens given soft prompt embeddings.

    This works with any HuggingFace causal LM that supports inputs_embeds.

    The model sees: [prompt_embeds | target_embeds]
    Loss is computed only on the target positions (prompt positions are masked with -100).

    Args:
        model: HuggingFace AutoModelForCausalLM (or compatible)
        target_tokens: Tokens to generate, shape (seq_len,)
        prompt_embeds: Soft prompt embeddings, shape (prompt_len, d) - can be float32

    Returns:
        Cross-entropy loss (scalar tensor, in float32 for stable gradients)
    """
    device = prompt_embeds.device

    # Get the embedding layer and its dtype
    embed_layer = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype

    # Convert prompt_embeds to model dtype for forward pass
    # (prompt_embeds may be float32 for optimization stability)
    prompt_embeds_cast = prompt_embeds.to(dtype=model_dtype)

    # Embed target tokens
    target_embeds = embed_layer(target_tokens.to(device))  # (seq_len, d)

    # Concatenate: prompt + target (both in model dtype)
    # Shape: (1, prompt_len + seq_len, d)
    full_embeds = torch.cat([prompt_embeds_cast, target_embeds], dim=0).unsqueeze(0)

    # Create labels for loss computation
    # For autoregressive LM: position i predicts token at position i+1
    # We want:
    #   - Prompt positions: -100 (ignored in loss)
    #   - Target positions: the actual target tokens (shifted appropriately)
    #
    # HuggingFace internally shifts labels, so labels[i] = token to predict at position i
    # Position (prompt_len - 1) should predict target[0]
    # Position (prompt_len) should predict target[1], etc.
    # Position (prompt_len + seq_len - 1) predicts nothing (or next token if continuing)

    prompt_len = prompt_embeds.shape[0]
    seq_len = target_tokens.shape[0]
    total_len = prompt_len + seq_len

    labels = torch.full((1, total_len), -100, dtype=torch.long, device=device)
    # The last prompt position predicts the first target token
    # So we place target_tokens starting at position (prompt_len - 1)
    labels[0, prompt_len - 1:prompt_len - 1 + seq_len] = target_tokens.to(device)

    # Forward pass with inputs_embeds
    outputs = model(inputs_embeds=full_embeds, labels=labels)

    return outputs.loss


# -----------------------------------------------------------------------------
# Scheduled Sampling + Unlikelihood Loss (Priorities 1 & 2)
# -----------------------------------------------------------------------------

def compute_epsilon(step: int, total_steps: int, config: OptimizationConfig) -> float:
    """
    Compute scheduled sampling epsilon for the current step.

    Epsilon controls the probability of using ground-truth vs model predictions:
    - ε = 1.0 means pure teacher forcing (always use ground truth)
    - ε = 0.0 means pure autoregressive (always use model predictions)

    Args:
        step: Current optimization step
        total_steps: Total number of steps
        config: Optimization config with schedule parameters

    Returns:
        Epsilon value in [ss_final_epsilon, ss_initial_epsilon]
    """
    if not config.use_scheduled_sampling:
        return 1.0  # Pure teacher forcing

    initial = config.ss_initial_epsilon
    final = config.ss_final_epsilon
    progress = step / max(total_steps - 1, 1)  # 0 to 1

    if config.ss_decay_type == "linear":
        # Linear decay: ε = initial - (initial - final) * progress
        epsilon = initial - (initial - final) * progress

    elif config.ss_decay_type == "exponential":
        # Exponential decay: ε = initial * (final/initial)^progress
        if final > 0:
            epsilon = initial * (final / initial) ** progress
        else:
            epsilon = initial * (1 - progress)

    elif config.ss_decay_type == "inverse_sigmoid":
        # Inverse sigmoid: ε = k / (k + exp(progress * k / 2))
        # Provides smooth transition with configurable steepness
        k = config.ss_decay_k
        # Map progress [0,1] to [-k/2, k/2] for sigmoid input
        x = (progress - 0.5) * k
        sigmoid = 1 / (1 + math.exp(-x))
        epsilon = initial - (initial - final) * sigmoid

    else:
        raise ValueError(f"Unknown decay type: {config.ss_decay_type}")

    return max(final, min(initial, epsilon))


def hf_scheduled_sampling_loss_fn(
    model: nn.Module,
    target_tokens: torch.Tensor,
    prompt_embeds: torch.Tensor,
    epsilon: float = 1.0,
    unlikelihood_alpha: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute loss with scheduled sampling and optional unlikelihood penalty.

    Scheduled sampling: At each position, with probability ε use ground-truth
    embedding, with probability (1-ε) use model's predicted embedding.
    This bridges the train-test gap by exposing the model to its own errors.

    Unlikelihood loss: Penalize the model for assigning high probability to
    tokens that have already appeared, reducing repetition loops.

    Args:
        model: HuggingFace AutoModelForCausalLM
        target_tokens: Target sequence, shape (seq_len,)
        prompt_embeds: Soft prompt embeddings, shape (prompt_len, d)
        epsilon: Probability of using ground-truth token (1.0 = pure teacher forcing)
        unlikelihood_alpha: Weight for unlikelihood loss (0.0 = disabled)

    Returns:
        (total_loss, ce_loss, ul_loss) - total, cross-entropy, and unlikelihood components
    """
    device = prompt_embeds.device
    embed_layer = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype

    prompt_embeds_cast = prompt_embeds.to(dtype=model_dtype)
    seq_len = target_tokens.shape[0]
    prompt_len = prompt_embeds.shape[0]

    # If pure teacher forcing (ε=1) and no unlikelihood, use fast path
    if epsilon >= 1.0 and unlikelihood_alpha <= 0:
        loss = hf_generator_loss_fn(model, target_tokens, prompt_embeds)
        return loss, loss, torch.tensor(0.0, device=device)

    # Build sequence with scheduled sampling
    # Start with prompt embeddings
    current_embeds = prompt_embeds_cast.unsqueeze(0)  # (1, prompt_len, d)

    ce_losses = []
    ul_losses = []
    seen_tokens = set()

    target_embeds = embed_layer(target_tokens.to(device))  # (seq_len, d)

    for t in range(seq_len):
        # Forward pass to get logits for position t
        with torch.no_grad() if t < seq_len - 1 else torch.enable_grad():
            # We need gradients for the final prediction, but intermediate
            # predictions are just for scheduled sampling decisions
            outputs = model(inputs_embeds=current_embeds)

        logits = outputs.logits[0, -1, :]  # (vocab_size,)
        target_token = target_tokens[t].item()

        # Cross-entropy loss for this position
        # Need to recompute with gradients for backprop
        if t == seq_len - 1 or epsilon < 1.0:
            # Full forward pass with gradients
            outputs_grad = model(inputs_embeds=current_embeds)
            logits_grad = outputs_grad.logits[0, -1, :]
            ce_loss_t = nn.functional.cross_entropy(
                logits_grad.unsqueeze(0),
                target_tokens[t:t+1].to(device)
            )
            ce_losses.append(ce_loss_t)

        # Unlikelihood loss: penalize probability of already-seen tokens
        if unlikelihood_alpha > 0 and seen_tokens:
            # Get probabilities
            probs = torch.softmax(logits_grad if 'logits_grad' in dir() else logits, dim=-1)
            # Sum log(1 - p) for seen tokens (we want to minimize -log(1-p), i.e., push p toward 0)
            ul_loss_t = torch.tensor(0.0, device=device, requires_grad=True)
            for seen_token in seen_tokens:
                if seen_token != target_token:  # Don't penalize if it's the correct next token
                    # -log(1 - p(seen_token)) - minimizing this pushes probability down
                    ul_loss_t = ul_loss_t - torch.log(1 - probs[seen_token] + 1e-10)
            ul_losses.append(ul_loss_t)

        seen_tokens.add(target_token)

        # Determine next input embedding (scheduled sampling)
        if t < seq_len - 1:
            if epsilon >= 1.0 or torch.rand(1).item() < epsilon:
                # Use ground-truth embedding
                next_embed = target_embeds[t:t+1, :].unsqueeze(0)  # (1, 1, d)
            else:
                # Use model's prediction embedding
                with torch.no_grad():
                    pred_token = torch.argmax(logits).item()
                    next_embed = embed_layer(
                        torch.tensor([[pred_token]], device=device)
                    )  # (1, 1, d)

            current_embeds = torch.cat([current_embeds, next_embed], dim=1)

    # Aggregate losses
    if ce_losses:
        ce_loss = torch.stack(ce_losses).mean()
    else:
        ce_loss = torch.tensor(0.0, device=device)

    if ul_losses:
        ul_loss = torch.stack(ul_losses).mean()
    else:
        ul_loss = torch.tensor(0.0, device=device)

    total_loss = ce_loss + unlikelihood_alpha * ul_loss

    return total_loss, ce_loss, ul_loss


def hf_semantic_loss_fn(
    model: nn.Module,
    target_tokens: torch.Tensor,
    prompt_embeds: torch.Tensor,
    temperature: float = 1.0,
    sentence_weight: float = 0.5,
    eos_token_id: Optional[int] = None,
    eos_penalty_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute semantic similarity loss at both token and sentence level.

    Token-level: At each position, compare expected predicted embedding to target embedding.
    Sentence-level: Compare mean hidden states of predicted vs target sequence.

    This allows semantically equivalent outputs (e.g., "London" vs "Greater London")
    to have low loss.

    Args:
        model: HuggingFace AutoModelForCausalLM
        target_tokens: Target sequence, shape (seq_len,) - should include EOS at end
        prompt_embeds: Soft prompt embeddings, shape (prompt_len, d)
        temperature: Softmax temperature (lower = sharper distribution)
        sentence_weight: Weight for sentence-level loss (0 = token only, 1 = sentence only)
        eos_token_id: EOS token ID for extra token penalty
        eos_penalty_weight: Weight for penalizing non-EOS after target (0 = disabled)

    Returns:
        (total_loss, token_loss, sentence_loss)
    """
    device = prompt_embeds.device
    embed_layer = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype

    prompt_embeds_cast = prompt_embeds.to(dtype=model_dtype)
    seq_len = target_tokens.shape[0]
    prompt_len = prompt_embeds.shape[0]

    # Get target embeddings
    target_embeds = embed_layer(target_tokens.to(device))  # (seq_len, d)

    # Get full embedding matrix for computing expected embeddings
    embed_matrix = embed_layer.weight  # (vocab_size, d)

    # Forward pass with teacher forcing - get hidden states too
    full_embeds = torch.cat([prompt_embeds_cast, target_embeds], dim=0).unsqueeze(0)
    outputs = model(inputs_embeds=full_embeds, output_hidden_states=True)

    # === Token-level loss ===
    # Get logits for each target position
    logits = outputs.logits[0, prompt_len-1:prompt_len-1+seq_len, :]  # (seq_len, vocab)

    # Compute softmax probabilities
    probs = torch.softmax(logits / temperature, dim=-1)  # (seq_len, vocab)

    # Compute expected embedding at each position: E[emb] = probs @ embed_matrix
    expected_embeds = probs @ embed_matrix  # (seq_len, d)

    # Normalize for cosine similarity
    expected_norm = expected_embeds / (expected_embeds.norm(dim=-1, keepdim=True) + 1e-8)
    target_input_norm = target_embeds / (target_embeds.norm(dim=-1, keepdim=True) + 1e-8)

    # Token-level cosine similarity
    token_cos_sim = (expected_norm * target_input_norm).sum(dim=-1)  # (seq_len,)
    token_loss = (1 - token_cos_sim).mean()

    # === Sentence-level loss ===
    # Get hidden states from the last layer
    hidden_states = outputs.hidden_states[-1]  # (1, prompt_len + seq_len, d)

    # Mean pool the hidden states for the generated positions
    # The positions where we predict target tokens
    pred_hidden = hidden_states[0, prompt_len-1:prompt_len-1+seq_len, :]  # (seq_len, d)
    pred_sentence = pred_hidden.mean(dim=0)  # (d,)

    # For target, we need to get hidden states from running target through the model
    # Use the hidden states at target positions (shifted by 1 from prediction positions)
    target_hidden = hidden_states[0, prompt_len:prompt_len+seq_len, :]  # (seq_len, d)
    target_sentence = target_hidden.mean(dim=0)  # (d,)

    # Sentence-level cosine similarity
    pred_sent_norm = pred_sentence / (pred_sentence.norm() + 1e-8)
    target_sent_norm = target_sentence / (target_sentence.norm() + 1e-8)
    sentence_cos_sim = (pred_sent_norm * target_sent_norm).sum()
    sentence_loss = 1 - sentence_cos_sim

    # Combined loss
    total_loss = (1 - sentence_weight) * token_loss + sentence_weight * sentence_loss

    # === EOS position penalty ===
    # Bidirectional: penalize both early EOS (during content) and late EOS (after content)
    if eos_penalty_weight > 0 and eos_token_id is not None:
        # Late EOS penalty: penalize non-EOS at position after target
        # Position (prompt_len + seq_len - 1) predicts what comes after last target token
        after_target_logits = outputs.logits[0, prompt_len + seq_len - 1, :]  # (vocab,)
        eos_target = torch.tensor([eos_token_id], device=device)
        late_eos_loss = nn.functional.cross_entropy(
            after_target_logits.unsqueeze(0),
            eos_target
        )

        # Early EOS penalty: penalize P(EOS) at content positions (before the final EOS)
        # Content positions are prompt_len-1 to prompt_len+seq_len-2 (predicting tokens 0 to seq_len-2)
        # We want P(EOS) to be LOW at these positions
        if seq_len > 1:  # Only if there's content before EOS
            content_logits = outputs.logits[0, prompt_len-1:prompt_len+seq_len-2, :]  # (seq_len-1, vocab)
            content_probs = torch.softmax(content_logits, dim=-1)
            early_eos_probs = content_probs[:, eos_token_id]  # P(EOS) at each content position
            # Penalize: -log(1 - P(EOS)) pushes P(EOS) toward 0
            early_eos_loss = -torch.log(1 - early_eos_probs + 1e-10).mean()
        else:
            early_eos_loss = torch.tensor(0.0, device=device)

        total_loss = total_loss + eos_penalty_weight * (late_eos_loss + early_eos_loss)

    return total_loss, token_loss, sentence_loss


def hf_scheduled_sampling_loss_fn_fast(
    model: nn.Module,
    target_tokens: torch.Tensor,
    prompt_embeds: torch.Tensor,
    epsilon: float = 1.0,
    unlikelihood_alpha: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast implementation of scheduled sampling loss using parallel computation.

    This version does a single forward pass and applies scheduled sampling
    by mixing embeddings, which is more efficient but an approximation.

    For epsilon < 1.0, it interpolates between GT and predicted embeddings
    rather than making discrete choices.

    Args:
        model: HuggingFace AutoModelForCausalLM
        target_tokens: Target sequence, shape (seq_len,)
        prompt_embeds: Soft prompt embeddings, shape (prompt_len, d)
        epsilon: Interpolation weight (1.0 = pure GT, 0.0 = pure predicted)
        unlikelihood_alpha: Weight for unlikelihood loss

    Returns:
        (total_loss, ce_loss, ul_loss)
    """
    device = prompt_embeds.device
    embed_layer = model.get_input_embeddings()
    model_dtype = next(model.parameters()).dtype

    prompt_embeds_cast = prompt_embeds.to(dtype=model_dtype)
    seq_len = target_tokens.shape[0]
    prompt_len = prompt_embeds.shape[0]

    # If pure teacher forcing and no unlikelihood, use standard loss
    if epsilon >= 1.0 and unlikelihood_alpha <= 0:
        loss = hf_generator_loss_fn(model, target_tokens, prompt_embeds)
        return loss, loss, torch.tensor(0.0, device=device)

    # Get target embeddings
    target_embeds = embed_layer(target_tokens.to(device))  # (seq_len, d)

    # First pass: teacher-forced to get predictions at each position
    full_embeds_tf = torch.cat([prompt_embeds_cast, target_embeds], dim=0).unsqueeze(0)

    with torch.no_grad():
        outputs_tf = model(inputs_embeds=full_embeds_tf)
        # Logits for predicting next token at each position
        # Position prompt_len-1 predicts target[0], etc.
        pred_logits = outputs_tf.logits[0, prompt_len-1:prompt_len-1+seq_len, :]  # (seq_len, vocab)
        pred_tokens = torch.argmax(pred_logits, dim=-1)  # (seq_len,)
        pred_embeds = embed_layer(pred_tokens)  # (seq_len, d)

    # Mix embeddings: ε * GT + (1-ε) * predicted
    mixed_embeds = epsilon * target_embeds + (1 - epsilon) * pred_embeds

    # Second pass: with mixed embeddings
    full_embeds_mixed = torch.cat([prompt_embeds_cast, mixed_embeds], dim=0).unsqueeze(0)

    # Create labels
    labels = torch.full((1, prompt_len + seq_len), -100, dtype=torch.long, device=device)
    labels[0, prompt_len-1:prompt_len-1+seq_len] = target_tokens.to(device)

    outputs = model(inputs_embeds=full_embeds_mixed, labels=labels)
    ce_loss = outputs.loss

    # Unlikelihood loss
    ul_loss = torch.tensor(0.0, device=device)
    if unlikelihood_alpha > 0:
        logits = outputs.logits[0, prompt_len-1:prompt_len-1+seq_len, :]  # (seq_len, vocab)
        probs = torch.softmax(logits, dim=-1)

        # For each position, penalize probability of tokens seen before that position
        for t in range(1, seq_len):
            seen = target_tokens[:t].unique()
            current_target = target_tokens[t].item()
            # Remove current target from penalty set (it's allowed to repeat if correct)
            seen = seen[seen != current_target]
            if len(seen) > 0:
                # -log(1 - p) for seen tokens
                seen_probs = probs[t, seen]
                ul_loss = ul_loss - torch.log(1 - seen_probs + 1e-10).sum()

        ul_loss = ul_loss / max(seq_len - 1, 1)

    total_loss = ce_loss + unlikelihood_alpha * ul_loss

    return total_loss, ce_loss, ul_loss


def create_hf_generator_loss_fn(model: nn.Module) -> Callable:
    """
    Create a loss function closure for a specific HuggingFace model.

    This is useful when passing the loss function to optimizers that
    expect a (model, target, prompt) -> loss signature.

    Args:
        model: HuggingFace AutoModelForCausalLM

    Returns:
        Callable with signature (model, target_tokens, prompt_embeds) -> loss
    """
    def loss_fn(model: nn.Module, target_tokens: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        return hf_generator_loss_fn(model, target_tokens, prompt_embeds)
    return loss_fn


# -----------------------------------------------------------------------------
# Batch processing utilities
# -----------------------------------------------------------------------------

def generate_targets_for_dataset(
    optimizer: ContentOptimizer,
    dataset: List[Tuple[torch.Tensor, torch.Tensor]],
    save_path: Optional[str] = None,
) -> List[OptimizationResult]:
    """
    Generate optimization targets for an entire dataset.

    Args:
        optimizer: Configured ContentOptimizer
        dataset: List of (Q_tokens, A_tokens) pairs
        save_path: Optional path to save results incrementally

    Returns:
        List of OptimizationResults (only converged ones if filtering)
    """
    results = []
    converged_count = 0

    for i, (Q_tokens, A_tokens) in enumerate(dataset):
        result = optimizer.optimize(Q_tokens, A_tokens)
        results.append(result)

        if result.converged:
            converged_count += 1

        if (i + 1) % 100 == 0:
            logger.info(
                f"Processed {i + 1}/{len(dataset)}, "
                f"converged: {converged_count}/{i + 1} "
                f"({100 * converged_count / (i + 1):.1f}%)"
            )

        # Incremental save
        if save_path and (i + 1) % 1000 == 0:
            torch.save(results, f"{save_path}_checkpoint_{i + 1}.pt")

    logger.info(
        f"Final: {converged_count}/{len(dataset)} converged "
        f"({100 * converged_count / len(dataset):.1f}%)"
    )

    if save_path:
        torch.save(results, save_path)

    return results
