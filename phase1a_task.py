"""
Phase 1a Task adapter for experiment-runner library.

This module provides a Task implementation that wraps the existing
ContentOptimizer for use with the experiment-runner distributed
execution framework.

Usage:
    from experiment_runner import SerialExecutor, DistributedExecutor
    from phase1a_task import Phase1aTask, Phase1aInput

    task = Phase1aTask(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", ...)
    executor = DistributedExecutor(workers=workers, storage=storage)
    result = executor.execute(task, inputs, job_id, artifact_dir)
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment_runner import Task, TaskContext, TaskResult

from phase1a_optimization import (
    ContentOptimizer,
    OptimizationConfig,
    OptimizationResult,
    create_mode_embeddings,
    hf_generator_loss_fn,
)


@dataclass
class Phase1aInput:
    """Input for a single Phase 1a optimization task."""

    index: int
    """Original index in the dataset."""

    q_text: str
    """Question text."""

    a_text: str
    """Answer text."""


class Phase1aTask(Task[Phase1aInput]):
    """Task implementation for Phase 1a joint Q-A optimization.

    Wraps ContentOptimizer for use with experiment-runner's distributed
    execution framework. Saves optimized embeddings as artifacts.
    """

    def __init__(
        self,
        model_name: str,
        config_dict: Dict[str, Any],
        add_eos: bool = True,
        save_embeddings: bool = True,
    ):
        """Initialize Phase1aTask.

        Args:
            model_name: HuggingFace model name
            config_dict: OptimizationConfig parameters as dict
            add_eos: Whether to append EOS to targets
            save_embeddings: Whether to save optimized embeddings as artifacts
        """
        self.model_name = model_name
        self.config_dict = config_dict
        self.add_eos = add_eos
        self.save_embeddings = save_embeddings

        # Will be initialized in setup()
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.optimizer: Optional[ContentOptimizer] = None
        self.Q_mode: Optional[torch.Tensor] = None
        self.A_mode: Optional[torch.Tensor] = None
        self.device: Optional[torch.device] = None

    def setup(self, ctx: TaskContext) -> None:
        """Load model and initialize optimizer on the assigned device."""
        self.device = torch.device(ctx.device)

        print(f"[Phase1aTask] Loading model on {ctx.device}...", flush=True)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Map GPU ID to device_map
        if "cuda" in ctx.device:
            device_map = {"": ctx.gpu_id}
        else:
            device_map = None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if "cuda" in ctx.device else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Create mode embeddings
        q_token_id = self.tokenizer.encode("Q", add_special_tokens=False)[0]
        a_token_id = self.tokenizer.encode("A", add_special_tokens=False)[0]
        self.Q_mode, self.A_mode = create_mode_embeddings(
            self.model, q_token_id, a_token_id
        )
        self.Q_mode = self.Q_mode.to(self.device)
        self.A_mode = self.A_mode.to(self.device)

        # Create optimization config
        # Ensure d is set from model
        config_dict = dict(self.config_dict)
        config_dict["d"] = self.model.config.hidden_size
        opt_config = OptimizationConfig(**config_dict)

        # Create optimizer
        self.optimizer = ContentOptimizer(
            generator=self.model,
            generator_loss_fn=hf_generator_loss_fn,
            Q_mode=self.Q_mode,
            A_mode=self.A_mode,
            config=opt_config,
            device=self.device,
            eos_token_id=self.tokenizer.eos_token_id,
            tokenizer=self.tokenizer,
        )

        print(f"[Phase1aTask] Setup complete on {ctx.device}", flush=True)

    def run(self, input_data: Phase1aInput, ctx: TaskContext) -> TaskResult:
        """Run optimization for a single Q-A pair."""
        start_time = time.perf_counter()

        # Tokenize the pair
        q_tokens = self.tokenizer.encode(
            input_data.q_text, return_tensors="pt", add_special_tokens=False
        ).squeeze(0).to(self.device)
        a_tokens = self.tokenizer.encode(
            input_data.a_text, return_tensors="pt", add_special_tokens=False
        ).squeeze(0).to(self.device)

        # Append EOS if configured
        if self.add_eos and self.tokenizer.eos_token_id is not None:
            eos_tensor = torch.tensor([self.tokenizer.eos_token_id], device=self.device)
            q_tokens = torch.cat([q_tokens, eos_tensor])
            a_tokens = torch.cat([a_tokens, eos_tensor])

        # Run optimization
        result = self.optimizer.optimize(q_tokens, a_tokens)

        # Build artifact paths
        artifact_paths: Dict[str, Path] = {}

        # Save optimized embeddings if configured and converged
        if self.save_embeddings and result.converged:
            embedding_path = ctx.save_artifact("content", result.content.cpu())
            artifact_paths["content"] = embedding_path

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Log result
        status = "✓" if result.converged else "✗"
        print(
            f"  {status} [{input_data.index}] loss={result.loss:.6f} "
            f"(Q: {result.loss_q:.6f}, A: {result.loss_a:.6f}) "
            f"restarts: {result.num_restarts_tried}",
            flush=True,
        )

        return TaskResult(
            task_id=f"{ctx.job_id}_{ctx.task_index}",
            index=ctx.task_index,
            success=True,  # Task completed (converged or not)
            metrics={
                "original_index": input_data.index,
                "loss": result.loss,
                "loss_q": result.loss_q,
                "loss_a": result.loss_a,
                "converged": result.converged,
                "num_restarts_tried": result.num_restarts_tried,
                "steps_to_converge": result.steps_to_converge,
            },
            artifact_paths=artifact_paths,
            execution_time_ms=elapsed_ms,
        )

    def teardown(self, ctx: TaskContext) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
        torch.cuda.empty_cache()

    def get_config(self) -> Dict[str, Any]:
        """Return serializable configuration."""
        return {
            "model_name": self.model_name,
            "config_dict": self.config_dict,
            "add_eos": self.add_eos,
            "save_embeddings": self.save_embeddings,
        }


def create_inputs_from_pairs(
    pairs: List[Tuple[str, str]],
    start_index: int = 0,
) -> List[Phase1aInput]:
    """Convert Q-A string pairs to Phase1aInput objects.

    Args:
        pairs: List of (question, answer) string tuples
        start_index: Starting index for numbering

    Returns:
        List of Phase1aInput objects
    """
    return [
        Phase1aInput(index=start_index + i, q_text=q, a_text=a)
        for i, (q, a) in enumerate(pairs)
    ]
