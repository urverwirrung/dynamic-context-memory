"""
Phase 1a: STM Target Generation via Optimization

This module implements the core optimization loop for finding content embeddings
that minimize Generator loss on (Q, A) pairs. Each successful optimization
produces a training target for the Diffuser.

Reference: dcm-plan.md, Part III > Target Generation via Optimization
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for content embedding optimization."""
    K: int                          # Number of content embedding slots
    d: int                          # Embedding dimension (must match Generator)
    num_restarts: int = 5           # Independent optimization runs per (Q, A)
    num_steps: int = 500            # Gradient steps per run
    lr: float = 0.01                # Initial learning rate
    init_scale: float = 0.02        # Scale for random initialization
    loss_threshold: float = 2.0     # Max acceptable loss for a valid target
    use_scheduler: bool = True      # Use cosine annealing LR schedule
    grad_clip: Optional[float] = 1.0  # Gradient clipping (None to disable)


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
    ):
        """
        Args:
            generator: Frozen language model
            generator_loss_fn: Function(model, target_tokens, prompt_embeds) -> loss
            Q_mode: Fixed mode embedding for question generation, shape (1, d)
            A_mode: Fixed mode embedding for answer generation, shape (1, d)
            config: Optimization hyperparameters
            device: Torch device
        """
        self.generator = generator
        self.generator_loss_fn = generator_loss_fn
        self.Q_mode = Q_mode
        self.A_mode = A_mode
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Ensure generator is frozen
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    def _initialize_content(self) -> torch.Tensor:
        """Initialize content embeddings with small random values."""
        content = torch.randn(
            self.config.K,
            self.config.d,
            device=self.device
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

        Returns:
            total_loss, loss_q, loss_a
        """
        # Construct prompt embeddings: [mode, content...]
        prompt_q = torch.cat([self.Q_mode, content], dim=0)  # (1+K, d)
        prompt_a = torch.cat([self.A_mode, content], dim=0)  # (1+K, d)

        # Compute reconstruction losses
        loss_q = self.generator_loss_fn(self.generator, Q_tokens, prompt_q)
        loss_a = self.generator_loss_fn(self.generator, A_tokens, prompt_a)

        return loss_q + loss_a, loss_q, loss_a

    def _single_run(
        self,
        Q_tokens: torch.Tensor,
        A_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, List[float]]:
        """
        Single optimization run from random initialization.

        Returns:
            best_content, best_loss, loss_history
        """
        content = self._initialize_content()
        optimizer = torch.optim.Adam([content], lr=self.config.lr)

        scheduler = None
        if self.config.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.num_steps
            )

        loss_history = []
        best_content = content.detach().clone()
        best_loss = float('inf')

        for step in range(self.config.num_steps):
            optimizer.zero_grad()

            loss, loss_q, loss_a = self._compute_loss(content, Q_tokens, A_tokens)

            loss.backward()

            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([content], self.config.grad_clip)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_content = content.detach().clone()

        return best_content, best_loss, loss_history

    def optimize(
        self,
        Q_tokens: torch.Tensor,
        A_tokens: torch.Tensor,
    ) -> OptimizationResult:
        """
        Find optimal content embeddings for a (Q, A) pair.

        Uses multiple restarts and returns the best result.

        Args:
            Q_tokens: Tokenized question, shape (seq_len_q,)
            A_tokens: Tokenized answer, shape (seq_len_a,)

        Returns:
            OptimizationResult with best content embeddings and metadata
        """
        best_content = None
        best_loss = float('inf')
        best_history = []

        for restart in range(self.config.num_restarts):
            content, loss, history = self._single_run(Q_tokens, A_tokens)

            if loss < best_loss:
                best_loss = loss
                best_content = content
                best_history = history

                # Early exit if we've found a good solution
                if loss < self.config.loss_threshold * 0.5:
                    logger.debug(f"Early exit at restart {restart + 1}")
                    break

        # Compute final component losses for diagnostics
        with torch.no_grad():
            _, loss_q, loss_a = self._compute_loss(best_content, Q_tokens, A_tokens)

        return OptimizationResult(
            content=best_content,
            loss=best_loss,
            loss_q=loss_q.item(),
            loss_a=loss_a.item(),
            converged=best_loss < self.config.loss_threshold,
            num_restarts_tried=restart + 1,
            loss_history=best_history,
        )


# -----------------------------------------------------------------------------
# Single-target optimization (diagnostic)
# -----------------------------------------------------------------------------

@dataclass
class SingleTargetResult:
    """Result of single-target optimization diagnostic."""
    content: torch.Tensor           # Shape (K, d) - the optimized embeddings
    loss: float                     # Final loss
    converged: bool                 # Whether loss < threshold
    num_restarts_tried: int
    loss_history: List[float]


class SingleTargetOptimizer:
    """
    Diagnostic: optimize content for a single target sequence.

    This is a simpler test than joint (Q, A) optimization. If single-target
    fails, the joint objective won't work either. Use this to validate that
    the Generator's embedding space is addressable before attempting the
    harder dual-mode optimization.

    Objective: c* = argmin_c [ L_gen(target | c) ]

    No mode embedding — just raw content → target reconstruction.
    """

    def __init__(
        self,
        generator: nn.Module,
        generator_loss_fn: Callable,
        config: OptimizationConfig,
        device: torch.device = None,
    ):
        self.generator = generator
        self.generator_loss_fn = generator_loss_fn
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    def _initialize_content(self) -> torch.Tensor:
        content = torch.randn(
            self.config.K,
            self.config.d,
            device=self.device
        ) * self.config.init_scale
        content.requires_grad_(True)
        return content

    def _single_run(
        self,
        target_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, List[float]]:
        content = self._initialize_content()
        optimizer = torch.optim.Adam([content], lr=self.config.lr)

        scheduler = None
        if self.config.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.num_steps
            )

        loss_history = []
        best_content = content.detach().clone()
        best_loss = float('inf')

        for step in range(self.config.num_steps):
            optimizer.zero_grad()

            loss = self.generator_loss_fn(self.generator, target_tokens, content)
            loss.backward()

            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([content], self.config.grad_clip)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_content = content.detach().clone()

        return best_content, best_loss, loss_history

    def optimize(self, target_tokens: torch.Tensor) -> SingleTargetResult:
        """
        Find content embeddings that reconstruct a single target sequence.

        Args:
            target_tokens: Tokenized target, shape (seq_len,)

        Returns:
            SingleTargetResult with optimized embeddings and diagnostics
        """
        best_content = None
        best_loss = float('inf')
        best_history = []

        for restart in range(self.config.num_restarts):
            content, loss, history = self._single_run(target_tokens)

            if loss < best_loss:
                best_loss = loss
                best_content = content
                best_history = history

                if loss < self.config.loss_threshold * 0.5:
                    break

        return SingleTargetResult(
            content=best_content,
            loss=best_loss,
            converged=best_loss < self.config.loss_threshold,
            num_restarts_tried=restart + 1,
            loss_history=best_history,
        )

    def run_diagnostic(
        self,
        targets: List[torch.Tensor],
        names: Optional[List[str]] = None,
    ) -> dict:
        """
        Run single-target optimization on multiple sequences and report statistics.

        Use this to assess feasibility before joint optimization.

        Args:
            targets: List of tokenized sequences
            names: Optional names for reporting

        Returns:
            Dict with convergence stats and per-target results
        """
        results = []
        for i, target in enumerate(targets):
            result = self.optimize(target)
            results.append(result)
            name = names[i] if names else f"target_{i}"
            status = "✓" if result.converged else "✗"
            logger.info(f"{status} {name}: loss={result.loss:.4f}")

        converged = sum(1 for r in results if r.converged)
        losses = [r.loss for r in results]

        return {
            "converged": converged,
            "total": len(targets),
            "convergence_rate": converged / len(targets),
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

    with torch.no_grad():
        Q_mode = embed(torch.tensor([q_token_id])).detach()  # (1, d)
        A_mode = embed(torch.tensor([a_token_id])).detach()  # (1, d)

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
        prompt_embeds: Soft prompt embeddings, shape (prompt_len, d)

    Returns:
        Cross-entropy loss (scalar tensor)
    """
    device = prompt_embeds.device

    # Get the embedding layer
    embed_layer = model.get_input_embeddings()

    # Embed target tokens
    target_embeds = embed_layer(target_tokens.to(device))  # (seq_len, d)

    # Concatenate: prompt + target
    # Shape: (1, prompt_len + seq_len, d)
    full_embeds = torch.cat([prompt_embeds, target_embeds], dim=0).unsqueeze(0)

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
