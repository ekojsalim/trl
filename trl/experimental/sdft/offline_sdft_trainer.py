# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import types
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import accelerate
import torch
from datasets import Dataset, IterableDataset
from packaging.version import Version
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, TrainerCallback
from transformers.utils import is_peft_available

from ...data_utils import maybe_apply_chat_template
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import get_config_model_id, pad
from ..self_distillation.teacher_context import extract_last_user_text
from .offline_sdft_config import DEFAULT_OFFLINE_SDFT_TEACHER_PROMPT_TEMPLATE, OfflineSDFTConfig
from .sdft_trainer import SDFTTrainer


if is_peft_available():
    from peft import PeftConfig
    from peft.peft_model import PeftModel


@dataclass
class _OfflineSDFTChunkedOutput:
    topk_log_probs: torch.Tensor
    topk_indices: torch.Tensor | None = None
    per_token_logps: torch.Tensor | None = None


class _OfflineSDFTStudentTopKProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: torch.Tensor | None,
        completion_ids: torch.Tensor,
        topk: int,
        logit_scale: float,
        final_logit_softcapping: float | None,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = _offline_sdft_project_logits(
            hidden_states,
            lm_head_weight,
            lm_head_bias,
            logit_scale,
            final_logit_softcapping,
            temperature,
        )
        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)
        topk_log_probs = topk_logits - logsumexp
        per_token_logps = (torch.gather(logits, dim=-1, index=completion_ids.unsqueeze(-1)) - logsumexp).squeeze(-1)

        if lm_head_bias is None:
            ctx.save_for_backward(hidden_states, lm_head_weight, completion_ids, topk_indices)
        else:
            ctx.save_for_backward(hidden_states, lm_head_weight, lm_head_bias, completion_ids, topk_indices)
        ctx.has_bias = lm_head_bias is not None
        ctx.logit_scale = logit_scale
        ctx.final_logit_softcapping = final_logit_softcapping
        ctx.temperature = temperature
        return topk_log_probs, topk_indices, per_token_logps

    @staticmethod
    def backward(ctx, grad_topk_log_probs, grad_topk_indices, grad_per_token_logps):
        if ctx.has_bias:
            hidden_states, lm_head_weight, lm_head_bias, completion_ids, topk_indices = ctx.saved_tensors
        else:
            hidden_states, lm_head_weight, completion_ids, topk_indices = ctx.saved_tensors
            lm_head_bias = None

        hidden_states_f = hidden_states.float()
        lm_head_weight_f = lm_head_weight.float()
        pre_scale_logits = hidden_states_f @ lm_head_weight_f.t()
        if lm_head_bias is not None:
            pre_scale_logits = pre_scale_logits + lm_head_bias.float()

        scaled_logits = pre_scale_logits * ctx.logit_scale if ctx.logit_scale != 1.0 else pre_scale_logits
        if ctx.final_logit_softcapping is not None:
            softcap_input = scaled_logits / ctx.final_logit_softcapping
            projected_logits = ctx.final_logit_softcapping * torch.tanh(softcap_input)
            softcap_grad = 1 - torch.tanh(softcap_input).pow(2)
        else:
            projected_logits = scaled_logits
            softcap_grad = None
        logits = projected_logits / ctx.temperature
        probs = torch.softmax(logits, dim=-1)

        grad_logits = torch.zeros_like(logits)
        if grad_topk_log_probs is not None:
            grad_topk_log_probs = grad_topk_log_probs.float()
            grad_logits.scatter_add_(dim=-1, index=topk_indices, src=grad_topk_log_probs)
            grad_logits = grad_logits - probs * grad_topk_log_probs.sum(dim=-1, keepdim=True)
        if grad_per_token_logps is not None:
            grad_per_token_logps = grad_per_token_logps.float().unsqueeze(-1)
            grad_logits.scatter_add_(dim=-1, index=completion_ids.unsqueeze(-1), src=grad_per_token_logps)
            grad_logits = grad_logits - probs * grad_per_token_logps

        grad_projected_logits = grad_logits / ctx.temperature
        if softcap_grad is not None:
            grad_scaled_logits = grad_projected_logits * softcap_grad
        else:
            grad_scaled_logits = grad_projected_logits
        grad_pre_scale_logits = grad_scaled_logits * ctx.logit_scale

        needs_hidden_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        grad_hidden_states = grad_pre_scale_logits @ lm_head_weight_f if needs_hidden_grad else None
        grad_lm_head_weight = grad_pre_scale_logits.t() @ hidden_states_f if needs_weight_grad else None
        grad_lm_head_bias = grad_pre_scale_logits.sum(dim=0) if needs_bias_grad and lm_head_bias is not None else None

        return (
            grad_hidden_states.to(hidden_states.dtype) if grad_hidden_states is not None else None,
            grad_lm_head_weight.to(lm_head_weight.dtype) if grad_lm_head_weight is not None else None,
            grad_lm_head_bias.to(lm_head_bias.dtype) if grad_lm_head_bias is not None else None,
            None,
            None,
            None,
            None,
            None,
        )


def _offline_sdft_project_logits(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    logit_scale: float,
    final_logit_softcapping: float | None,
    temperature: float,
) -> torch.Tensor:
    logits = hidden_states.float() @ lm_head_weight.float().t()
    if lm_head_bias is not None:
        logits = logits + lm_head_bias.float()
    if logit_scale != 1.0:
        logits = logits * logit_scale
    if final_logit_softcapping is not None:
        logits = final_logit_softcapping * torch.tanh(logits / final_logit_softcapping)
    return logits / temperature


def _offline_sdft_student_chunk(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    completion_ids: torch.Tensor,
    topk: int,
    logit_scale: float,
    final_logit_softcapping: float | None,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = _offline_sdft_project_logits(
        hidden_states,
        lm_head_weight,
        lm_head_bias,
        logit_scale,
        final_logit_softcapping,
        temperature,
    )
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)
    topk_log_probs = topk_logits - logsumexp
    per_token_logps = (torch.gather(logits, dim=-1, index=completion_ids.unsqueeze(-1)) - logsumexp).squeeze(-1)
    return topk_log_probs, topk_indices, per_token_logps


def _offline_sdft_teacher_chunk(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor | None,
    topk_indices: torch.Tensor,
    logit_scale: float,
    final_logit_softcapping: float | None,
    temperature: float,
) -> torch.Tensor:
    logits = _offline_sdft_project_logits(
        hidden_states,
        lm_head_weight,
        lm_head_bias,
        logit_scale,
        final_logit_softcapping,
        temperature,
    )
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
    return topk_logits - logsumexp


def _patch_offline_sdft_chunked_forward(
    model: nn.Module, chunk_size: int, temperature: float, force: bool = False
) -> None:
    if getattr(model, "_offline_sdft_chunked_forward_patched", False):
        if not force:
            return
        model.forward = types.MethodType(type(model).forward, model)
        model._offline_sdft_chunked_forward_patched = False

    model._offline_sdft_original_forward = model.forward
    text_config = getattr(model.config, "text_config", model.config)
    logit_scale = getattr(text_config, "logit_scale", 1.0)
    final_logit_softcapping = getattr(text_config, "final_logit_softcapping", None)

    def _offline_sdft_chunked_forward(
        self: nn.Module,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        offline_sdft_logits_to_keep: int | None = None,
        offline_sdft_response_mask: torch.Tensor | None = None,
        offline_sdft_completion_ids: torch.Tensor | None = None,
        offline_sdft_topk: int | None = None,
        offline_sdft_topk_indices: torch.Tensor | None = None,
        **kwargs,
    ):
        if offline_sdft_logits_to_keep is None:
            return self._offline_sdft_original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        kwargs.pop("use_cache", None)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs)
        hidden_states = outputs.last_hidden_state[:, :-1, :]
        hidden_states = hidden_states[:, -offline_sdft_logits_to_keep:, :].contiguous()
        batch_size, completion_length, hidden_size = hidden_states.shape

        if offline_sdft_response_mask is None:
            offline_sdft_response_mask = torch.ones(
                (batch_size, completion_length), device=hidden_states.device, dtype=torch.bool
            )
        flat_mask = offline_sdft_response_mask.reshape(-1).bool()
        hidden_states = hidden_states.reshape(-1, hidden_size)
        hidden_states = hidden_states[flat_mask]

        lm_head = self.get_output_embeddings()
        lm_head_weight = getattr(self, "_offline_sdft_lm_head_weight", lm_head.weight)
        lm_head_bias = getattr(self, "_offline_sdft_lm_head_bias", getattr(lm_head, "bias", None))

        if hidden_states.size(0) == 0:
            topk = offline_sdft_topk or 1
            topk = min(topk, lm_head_weight.size(0))
            topk_log_probs = lm_head_weight.new_zeros((batch_size, completion_length, topk))
            return _OfflineSDFTChunkedOutput(topk_log_probs=topk_log_probs)

        if offline_sdft_topk_indices is None:
            topk = min(offline_sdft_topk, lm_head_weight.size(0))
            completion_ids = offline_sdft_completion_ids.reshape(-1)[flat_mask]
            topk_log_probs_chunks = []
            topk_indices_chunks = []
            per_token_logps_chunks = []
            for start in range(0, hidden_states.size(0), chunk_size):
                if self.training and torch.is_grad_enabled():
                    topk_log_probs, topk_indices, per_token_logps = _OfflineSDFTStudentTopKProjection.apply(
                        hidden_states[start : start + chunk_size],
                        lm_head_weight,
                        lm_head_bias,
                        completion_ids[start : start + chunk_size],
                        topk,
                        logit_scale,
                        final_logit_softcapping,
                        temperature,
                    )
                else:
                    topk_log_probs, topk_indices, per_token_logps = _offline_sdft_student_chunk(
                        hidden_states[start : start + chunk_size],
                        lm_head_weight,
                        lm_head_bias,
                        completion_ids[start : start + chunk_size],
                        topk,
                        logit_scale,
                        final_logit_softcapping,
                        temperature,
                    )
                topk_log_probs_chunks.append(topk_log_probs)
                topk_indices_chunks.append(topk_indices)
                per_token_logps_chunks.append(per_token_logps)

            active_topk_log_probs = torch.cat(topk_log_probs_chunks, dim=0)
            active_topk_indices = torch.cat(topk_indices_chunks, dim=0)
            active_per_token_logps = torch.cat(per_token_logps_chunks, dim=0)

            topk_log_probs = active_topk_log_probs.new_zeros((flat_mask.numel(), topk))
            topk_indices = active_topk_indices.new_zeros((flat_mask.numel(), topk))
            per_token_logps = active_per_token_logps.new_zeros(flat_mask.numel())
            topk_log_probs[flat_mask] = active_topk_log_probs
            topk_indices[flat_mask] = active_topk_indices
            per_token_logps[flat_mask] = active_per_token_logps
            return _OfflineSDFTChunkedOutput(
                topk_log_probs=topk_log_probs.reshape(batch_size, completion_length, topk),
                topk_indices=topk_indices.reshape(batch_size, completion_length, topk),
                per_token_logps=per_token_logps.reshape(batch_size, completion_length),
            )

        topk = offline_sdft_topk_indices.size(-1)
        topk_indices = offline_sdft_topk_indices.reshape(-1, topk)[flat_mask]
        topk_log_probs_chunks = []
        for start in range(0, hidden_states.size(0), chunk_size):
            topk_log_probs = _offline_sdft_teacher_chunk(
                hidden_states[start : start + chunk_size],
                lm_head_weight,
                lm_head_bias,
                topk_indices[start : start + chunk_size],
                logit_scale,
                final_logit_softcapping,
                temperature,
            )
            topk_log_probs_chunks.append(topk_log_probs)

        active_topk_log_probs = torch.cat(topk_log_probs_chunks, dim=0)
        topk_log_probs = active_topk_log_probs.new_zeros((flat_mask.numel(), topk))
        topk_log_probs[flat_mask] = active_topk_log_probs
        return _OfflineSDFTChunkedOutput(
            topk_log_probs=topk_log_probs.reshape(batch_size, completion_length, topk),
        )

    model.forward = types.MethodType(_offline_sdft_chunked_forward, model)
    model._offline_sdft_chunked_forward_patched = True


def _stringify_teacher_context(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        chunks = []
        for message in value:
            content = message.get("content", "")
            if isinstance(content, list):
                text = " ".join(part.get("text", "") for part in content if part.get("type") == "text")
            else:
                text = str(content)
            if text:
                chunks.append(text)
        return "\n".join(chunks)
    return str(value)


def render_offline_sdft_teacher_prompt(
    prompt: Any,
    feedback: Any,
    correct_solution: Any,
    teacher_prompt_template: str = DEFAULT_OFFLINE_SDFT_TEACHER_PROMPT_TEMPLATE,
) -> Any:
    """
    Render a privileged offline SDFT teacher prompt from raw prompt, feedback, and correct solution fields.

    Args:
        prompt (`str` or `list[dict]`):
            Student-facing prompt.
        feedback (`Any`):
            Judge or environment feedback for the unsuccessful rollout.
        correct_solution (`Any`):
            Verified solution or answer used as privileged teacher context.
        teacher_prompt_template (`str`, *optional*):
            Template with `{prompt}`, `{feedback}`, and `{correct_solution}` placeholders.

    Returns:
        `str` or `list[dict]`: Rendered teacher prompt matching the input prompt format.
    """
    if isinstance(prompt, list):
        prompt_text = extract_last_user_text(prompt)
        teacher_text = teacher_prompt_template.format(
            prompt=prompt_text,
            feedback=_stringify_teacher_context(feedback),
            correct_solution=_stringify_teacher_context(correct_solution),
        )
        return prompt[:-1] + [{"role": "user", "content": teacher_text}]
    return teacher_prompt_template.format(
        prompt=prompt,
        feedback=_stringify_teacher_context(feedback),
        correct_solution=_stringify_teacher_context(correct_solution),
    )


class OfflineSDFTTrainer(SDFTTrainer):
    """
    Trainer for offline/static-rollout self-distillation.

    Offline SDFT trains on fixed completions from the dataset. At each prefix of the fixed completion, it distills the
    model conditioned on a privileged teacher prompt into the model conditioned on the ordinary student prompt.
    """

    _tag_names = ["trl", "offline-sdft"]
    _name = "Offline SDFT"
    config_cls = OfflineSDFTConfig

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        args: OfflineSDFTConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: PeftConfig | None = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            args = OfflineSDFTConfig(output_dir=f"{model_name.split('/')[-1]}-OfflineSDFT")
        if args.sync_ref_model:
            raise ValueError("OfflineSDFTTrainer does not support `sync_ref_model=True`.")

        self._offline_sdft_uses_patched_forward = False
        self._offline_sdft_args = args

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

    def _should_patch_chunked_topk_distillation(self) -> bool:
        return (
            self._offline_sdft_args.distillation_chunk_size is not None
            and self._offline_sdft_args.distillation_topk is not None
        )

    def _patch_chunked_topk_distillation_model(self, model, force: bool = False):
        if not self._should_patch_chunked_topk_distillation():
            return model

        target_model = model.get_base_model() if is_peft_available() and isinstance(model, PeftModel) else model
        _patch_offline_sdft_chunked_forward(
            target_model,
            chunk_size=self._offline_sdft_args.distillation_chunk_size,
            temperature=self.temperature,
            force=force,
        )
        self._offline_sdft_uses_patched_forward = True
        return model

    def _prepare_model_for_trainer_wrapping(self, model):
        return self._patch_chunked_topk_distillation_model(model)

    def _get_offline_sdft_patched_model(self, model):
        unwrapped_model = self.accelerator.unwrap_model(model)
        if is_peft_available() and isinstance(unwrapped_model, PeftModel):
            return unwrapped_model.get_base_model()
        return unwrapped_model

    @contextmanager
    def _cache_chunked_lm_head_for_fsdp2(self, model):
        if not self._offline_sdft_uses_patched_forward:
            yield
            return
        if not (
            Version(accelerate.__version__) >= Version("1.6.0")
            and self.accelerator.state.is_fsdp2
            and self.accelerator.state.fsdp_plugin.reshard_after_forward
        ):
            yield
            return

        target_model = self._get_offline_sdft_patched_model(model)
        lm_head = target_model.get_output_embeddings()
        if lm_head.weight.requires_grad or (lm_head.bias is not None and lm_head.bias.requires_grad):
            raise NotImplementedError(
                "OfflineSDFTTrainer's chunked top-k path with FSDP2 `reshard_after_forward=True` currently supports "
                "only frozen LM-head parameters. This matches the PEFT setting where adapters are trainable and the "
                "base LM head is frozen."
            )

        from torch.distributed.tensor import DTensor

        weight = lm_head.weight.full_tensor() if isinstance(lm_head.weight, DTensor) else lm_head.weight
        bias = None
        if lm_head.bias is not None:
            bias = lm_head.bias.full_tensor() if isinstance(lm_head.bias, DTensor) else lm_head.bias

        target_model._offline_sdft_lm_head_weight = weight
        target_model._offline_sdft_lm_head_bias = bias
        try:
            yield
        finally:
            del target_model._offline_sdft_lm_head_weight
            del target_model._offline_sdft_lm_head_bias

    def _get_zero3_lm_head_gather_ctx(self, model):
        if not self._offline_sdft_uses_patched_forward:
            return nullcontext()

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        if deepspeed_plugin is None or deepspeed_plugin.zero_stage != 3:
            return nullcontext()

        import deepspeed

        target_model = self._get_offline_sdft_patched_model(model)
        lm_head = target_model.get_output_embeddings()
        params = [lm_head.weight]
        if lm_head.bias is not None:
            params.append(lm_head.bias)
        return deepspeed.zero.GatheredParameters(params, modifier_rank=None)

    @contextmanager
    def _get_chunked_lm_head_gather_ctx(self, model):
        with self._get_zero3_lm_head_gather_ctx(model):
            with self._cache_chunked_lm_head_for_fsdp2(model):
                yield

    def training_step(self, model, inputs, num_items_in_batch):
        with self._get_chunked_lm_head_gather_ctx(model):
            return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        super().log({**logs, **metrics}, start_time)
        self._metrics[mode].clear()

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            columns = [
                self.args.student_prompt_column,
                self.args.teacher_prompt_column,
                self.args.completion_column,
                self.args.prompt_column,
                self.args.feedback_column,
                self.args.correct_solution_column,
            ]
            if self.args.old_per_token_logps_column is not None:
                columns.append(self.args.old_per_token_logps_column)
            self._signature_columns = columns

    def get_train_dataloader(self):
        return _BaseTrainer.get_train_dataloader(self)

    def _get_train_sampler(self, dataset=None):
        return _BaseTrainer._get_train_sampler(self, dataset)

    def _get_eval_sampler(self, eval_dataset):
        return _BaseTrainer._get_eval_sampler(self, eval_dataset)

    def _prepare_inputs(self, inputs):
        if isinstance(inputs, dict) and "prompt_ids" in inputs:
            return _BaseTrainer._prepare_inputs(self, inputs)
        if isinstance(inputs, dict):
            inputs = [inputs]
        return self._build_static_batch(inputs)

    def _normalize_example(self, example: dict[str, Any]) -> tuple[Any, Any, Any]:
        if self.args.completion_column not in example:
            raise KeyError(f"Expected dataset column `{self.args.completion_column}` for OfflineSDFTTrainer.")

        if self.args.teacher_prompt_column in example:
            teacher_prompt = example[self.args.teacher_prompt_column]
            if self.args.student_prompt_column in example:
                student_prompt = example[self.args.student_prompt_column]
            elif self.args.prompt_column in example:
                student_prompt = example[self.args.prompt_column]
            else:
                raise KeyError(
                    f"Expected `{self.args.student_prompt_column}` or `{self.args.prompt_column}` when "
                    f"`{self.args.teacher_prompt_column}` is provided."
                )
        elif (
            self.args.prompt_column in example
            and self.args.feedback_column in example
            and self.args.correct_solution_column in example
        ):
            student_prompt = example[self.args.prompt_column]
            teacher_prompt = render_offline_sdft_teacher_prompt(
                student_prompt,
                example[self.args.feedback_column],
                example[self.args.correct_solution_column],
                self.args.teacher_prompt_template,
            )
        else:
            raise KeyError(
                "OfflineSDFTTrainer expects either "
                f"`{self.args.student_prompt_column}`, `{self.args.teacher_prompt_column}`, and "
                f"`{self.args.completion_column}` columns, or raw `{self.args.prompt_column}`, "
                f"`{self.args.feedback_column}`, `{self.args.correct_solution_column}`, and "
                f"`{self.args.completion_column}` columns."
            )

        return student_prompt, teacher_prompt, example[self.args.completion_column]

    def _completion_to_text(self, prompt: Any, completion: Any) -> str:
        is_chat_prompt = isinstance(prompt, list)
        if is_chat_prompt:
            completion_messages = completion
            if isinstance(completion, str):
                completion_messages = [{"role": "assistant", "content": completion}]
            formatted = maybe_apply_chat_template(
                {"prompt": prompt, "completion": completion_messages},
                self.processing_class,
                **self.chat_template_kwargs,
            )
            completion_text = formatted["completion"]
        else:
            completion_text = _stringify_teacher_context(completion)

        if self.args.append_eos_token and self._tokenizer.eos_token is not None:
            if not completion_text.endswith(self._tokenizer.eos_token):
                completion_text = completion_text + self._tokenizer.eos_token
        return completion_text

    def _tokenize_static_completions(
        self,
        prompts: list[Any],
        completions: list[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        completion_text = [
            self._completion_to_text(prompt, completion)
            for prompt, completion in zip(prompts, completions, strict=True)
        ]
        truncation_side = self._tokenizer.truncation_side
        self._tokenizer.truncation_side = "right"
        try:
            completion_inputs = self.processing_class(
                text=completion_text,
                return_tensors="pt",
                padding=True,
                padding_side="right",
                max_length=self.max_completion_length,
                truncation=self.max_completion_length is not None,
                add_special_tokens=False,
            )
        finally:
            self._tokenizer.truncation_side = truncation_side
        completion_inputs = _BaseTrainer._prepare_inputs(self, completion_inputs)
        completion_ids = [
            c[m].tolist()
            for c, m in zip(
                completion_inputs["input_ids"],
                completion_inputs["attention_mask"].bool(),
                strict=True,
            )
        ]
        completion_ids = [torch.tensor(ids, device=self.accelerator.device) for ids in completion_ids]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        return (
            pad(completion_ids, padding_value=self._tokenizer.pad_token_id, padding_side="right"),
            pad(completion_mask, padding_value=0, padding_side="right"),
        )

    def _build_old_per_token_logps(
        self,
        inputs: list[dict[str, Any]],
        completion_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.args.old_per_token_logps_column is None:
            return None

        old_per_token_logps = []
        for example, mask in zip(inputs, completion_mask, strict=True):
            values = example[self.args.old_per_token_logps_column]
            completion_length = int(mask.sum().item())
            if len(values) < completion_length:
                raise ValueError(
                    f"`{self.args.old_per_token_logps_column}` must contain one value per active completion token; "
                    f"got {len(values)} values for {completion_length} tokens."
                )
            old_per_token_logps.append(
                torch.tensor(values[:completion_length], dtype=torch.float32, device=self.accelerator.device)
            )
        return pad(old_per_token_logps, padding_value=0.0, padding_side="right")

    def _build_static_batch(self, inputs: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        student_prompts = []
        teacher_prompts = []
        completions = []
        for example in inputs:
            student_prompt, teacher_prompt, completion = self._normalize_example(example)
            student_prompts.append(student_prompt)
            teacher_prompts.append(teacher_prompt)
            completions.append(completion)

        student_batch = self.prompt_tokenizer.tokenize_prompts(student_prompts)
        teacher_batch = self.prompt_tokenizer.tokenize_prompts(teacher_prompts)
        completion_ids, completion_mask = self._tokenize_static_completions(student_prompts, completions)

        teacher_input_ids = torch.cat([teacher_batch.prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_batch.prompt_mask, completion_mask], dim=1)
        old_per_token_logps = self._build_old_per_token_logps(inputs, completion_mask)

        self._dispatch_self_distillation_callback(
            "on_self_distillation_batch_prepared",
            old_per_token_logps=old_per_token_logps,
            prompt_ids=student_batch.prompt_ids,
            completion_ids=completion_ids,
        )

        output = {
            "prompt_ids": student_batch.prompt_ids,
            "prompt_mask": student_batch.prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        return output

    def _use_chunked_topk_distillation(self) -> bool:
        return (
            self.args.distillation_chunk_size is not None
            and self.args.distillation_topk is not None
            and (self.args.full_logit_distillation or self._allow_topk_without_full_logit_distillation())
        )

    @staticmethod
    def _entropy_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
        probs = log_probs.exp()
        return -(torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))).sum(dim=-1)

    def _log_self_distillation_metric(self, mode: str, metric_name: str, value: float) -> None:
        self._metrics[mode][f"offline_sdft/{metric_name}"].append(value)

    def _log_offline_sdft_masked_metric(
        self,
        mode: str,
        metric_name: str,
        values: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> None:
        values = values.detach().float()
        response_mask = response_mask.detach().to(values.dtype)
        mean_value = (values * response_mask).sum() / response_mask.sum().clamp(min=1.0)
        self._log_self_distillation_metric(mode, metric_name, self.accelerator.gather(mean_value).mean().item())

    def _compute_chunked_topk_distillation_loss(
        self,
        model,
        inputs: dict[str, Any],
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        logits_to_keep = completion_ids.size(1)

        student_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        student_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        student_output = model(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            use_cache=False,
            offline_sdft_logits_to_keep=logits_to_keep,
            offline_sdft_response_mask=response_mask,
            offline_sdft_completion_ids=completion_ids,
            offline_sdft_topk=self.args.distillation_topk,
        )

        teacher_model = self._get_teacher_model_for_self_distillation(model)
        with torch.no_grad(), self._get_teacher_context_for_self_distillation(model):
            teacher_output = teacher_model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
                use_cache=False,
                offline_sdft_logits_to_keep=logits_to_keep,
                offline_sdft_response_mask=response_mask,
                offline_sdft_topk_indices=student_output.topk_indices,
            )

        topk_student_log_probs = student_output.topk_log_probs
        topk_teacher_log_probs = teacher_output.topk_log_probs
        mode = "train" if model.training else "eval"
        self._log_offline_sdft_masked_metric(
            mode,
            "student_topk_mass",
            torch.exp(torch.logsumexp(topk_student_log_probs, dim=-1)),
            response_mask,
        )
        self._log_offline_sdft_masked_metric(
            mode,
            "teacher_topk_mass",
            torch.exp(torch.logsumexp(topk_teacher_log_probs, dim=-1)),
            response_mask,
        )

        if self.args.distillation_add_tail:
            topk_student_log_probs = self._add_tail(topk_student_log_probs)
            topk_teacher_log_probs = self._add_tail(topk_teacher_log_probs)
        else:
            topk_student_log_probs = self._renorm_topk_log_probs(topk_student_log_probs)
            topk_teacher_log_probs = self._renorm_topk_log_probs(topk_teacher_log_probs)
        self._log_offline_sdft_masked_metric(
            mode,
            "student_entropy",
            self._entropy_from_log_probs(topk_student_log_probs),
            response_mask,
        )
        self._log_offline_sdft_masked_metric(
            mode,
            "teacher_entropy",
            self._entropy_from_log_probs(topk_teacher_log_probs),
            response_mask,
        )

        per_token_loss = self._compute_divergence(
            topk_student_log_probs, topk_teacher_log_probs, self.args.distillation_alpha
        )
        old_per_token_logps = inputs.get("old_per_token_logps")
        if self.args.distillation_is_clip is not None and old_per_token_logps is not None:
            negative_approx_kl = (student_output.per_token_logps.detach() - old_per_token_logps).clamp(
                min=-20.0, max=20.0
            )
            is_ratio = torch.exp(negative_approx_kl)
            self._log_offline_sdft_masked_metric(mode, "is_ratio_mean", is_ratio, response_mask)
            self._log_offline_sdft_masked_metric(
                mode,
                "is_clipped_frac",
                (is_ratio > self.args.distillation_is_clip).float(),
                response_mask,
            )
            per_token_loss = per_token_loss * is_ratio.clamp(max=self.args.distillation_is_clip)

        return self._finish_chunked_self_distillation_loss(model, per_token_loss, response_mask)

    def _finish_chunked_self_distillation_loss(
        self,
        model,
        per_token_loss: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = self._aggregate_self_distillation_loss(per_token_loss, response_mask)

        mode = "train" if model.training else "eval"
        self._log_offline_sdft_masked_metric(mode, "distillation_loss", per_token_loss, response_mask)

        return loss

    def _compute_zero_self_distillation_loss(
        self,
        model,
        inputs: dict[str, Any],
    ) -> torch.Tensor:
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        student_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        student_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        model_inputs = {
            "input_ids": student_input_ids,
            "attention_mask": student_attention_mask,
            "use_cache": False,
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = 1

        logits = model(**model_inputs).logits
        loss = logits.float().sum() * 0.0

        mode = "train" if model.training else "eval"
        self._log_self_distillation_metric(mode, "distillation_loss", 0.0)
        return loss

    def _compute_self_distillation_loss(
        self,
        model,
        inputs: dict[str, Any],
    ) -> torch.Tensor:
        if not self._use_chunked_topk_distillation():
            return super()._compute_self_distillation_loss(model, inputs)

        completion_mask = inputs["completion_mask"]

        self_distillation_mask = inputs.get("self_distillation_mask")
        if self_distillation_mask is not None:
            response_mask = completion_mask * self_distillation_mask.unsqueeze(1)
        else:
            response_mask = completion_mask

        if response_mask.sum() == 0:
            return self._compute_zero_self_distillation_loss(model, inputs)

        if not self._offline_sdft_uses_patched_forward:
            raise RuntimeError("OfflineSDFTTrainer requires its chunked top-k forward patch before training.")
        return self._compute_chunked_topk_distillation_loss(model, inputs, response_mask)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The OfflineSDFTTrainer does not support returning outputs")

        if self.num_loss_tokens_to_skip > 0:
            inputs = dict(inputs)
            completion_mask = inputs["completion_mask"].clone()
            token_positions = torch.arange(completion_mask.size(1), device=completion_mask.device).unsqueeze(0)
            completion_mask = completion_mask * (token_positions >= self.num_loss_tokens_to_skip).long()
            inputs["completion_mask"] = completion_mask

        loss = self.args.distillation_weight * self._compute_self_distillation_loss(model, inputs)
        accumulation_scale = self.current_gradient_accumulation_steps if self.model.training else 1.0
        return loss / accumulation_scale
