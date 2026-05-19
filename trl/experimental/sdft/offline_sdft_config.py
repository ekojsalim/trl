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

from dataclasses import dataclass, field

from ..self_distillation.self_distillation_config import SelfDistillationConfig
from .sdft_config import SDFTConfig


DEFAULT_OFFLINE_SDFT_TEACHER_PROMPT_TEMPLATE = (
    "{prompt}\n\n"
    "Correct solution:\n{correct_solution}\n\n"
    "The following is judge feedback from an earlier unsuccessful attempt:\n"
    "{feedback}\n\n"
    "Correctly solve the original problem."
)


@dataclass
class OfflineSDFTConfig(SDFTConfig):
    r"""
    Configuration class for [`OfflineSDFTTrainer`].

    Offline SDFT consumes fixed rollout completions from the dataset and distills the model conditioned on a privileged
    teacher prompt into the model conditioned on the ordinary student prompt.

    Parameters:
        student_prompt_column (`str`, *optional*, defaults to `"student_prompt"`):
            Dataset column containing the student-facing prompt.
        teacher_prompt_column (`str`, *optional*, defaults to `"teacher_prompt"`):
            Dataset column containing the pre-rendered privileged teacher prompt.
        completion_column (`str`, *optional*, defaults to `"completion"`):
            Dataset column containing the fixed rollout completion.
        prompt_column (`str`, *optional*, defaults to `"prompt"`):
            Raw-schema prompt column used when `teacher_prompt_column` is absent.
        feedback_column (`str`, *optional*, defaults to `"feedback"`):
            Raw-schema feedback column used when `teacher_prompt_column` is absent.
        correct_solution_column (`str`, *optional*, defaults to `"correct_solution"`):
            Raw-schema correct solution column used when `teacher_prompt_column` is absent.
        teacher_prompt_template (`str`, *optional*):
            Template used to render a teacher prompt from raw `prompt`, `feedback`, and `correct_solution` columns.
        append_eos_token (`bool`, *optional*, defaults to `True`):
            Whether to append the tokenizer EOS token to static completion text when it is not already present.
        loss_on_completion_only (`bool`, *optional*, defaults to `True`):
            Whether to apply distillation only on completion tokens. Offline SDFT currently supports only `True`.
        old_per_token_logps_column (`str`, *optional*):
            Optional dataset column containing old per-token log probabilities for importance-sampling clipping.
        distillation_alpha (`float`, *optional*, defaults to `1.0`):
            Divergence interpolation coefficient. `1.0` corresponds to reverse KL, `KL(student || teacher)`.
        distillation_topk (`int` or `None`, *optional*, defaults to `100`):
            Number of student top tokens to keep for SDPO-style top-k distillation.
        distillation_add_tail (`bool`, *optional*, defaults to `True`):
            Whether to add a tail bucket for non-top-k probability mass.
        distillation_chunk_size (`int` or `None`, *optional*, defaults to `1024`):
            Number of completion tokens to process at a time for memory-efficient Liger-style top-k distillation. If
            `None`, the shared unchunked self-distillation path is used.
    """

    num_generations: int = field(
        default=1,
        metadata={"help": "Unused for offline SDFT; kept at 1 to disable online generation grouping."},
    )
    teacher_prompt_template: str = field(
        default=DEFAULT_OFFLINE_SDFT_TEACHER_PROMPT_TEMPLATE,
        metadata={"help": "Template used to render teacher prompts from raw prompt, feedback, and solution columns."},
    )
    student_prompt_column: str = field(
        default="student_prompt",
        metadata={"help": "Dataset column containing the student prompt."},
    )
    teacher_prompt_column: str = field(
        default="teacher_prompt",
        metadata={"help": "Dataset column containing the privileged teacher prompt."},
    )
    completion_column: str = field(
        default="completion",
        metadata={"help": "Dataset column containing the fixed rollout completion."},
    )
    prompt_column: str = field(
        default="prompt",
        metadata={"help": "Raw-schema prompt column used when the teacher prompt column is absent."},
    )
    feedback_column: str = field(
        default="feedback",
        metadata={"help": "Raw-schema feedback column used when the teacher prompt column is absent."},
    )
    correct_solution_column: str = field(
        default="correct_solution",
        metadata={"help": "Raw-schema correct solution column used when the teacher prompt column is absent."},
    )
    append_eos_token: bool = field(
        default=True,
        metadata={"help": "Whether to append the tokenizer EOS token to static completions."},
    )
    loss_on_completion_only: bool = field(
        default=True,
        metadata={"help": "Whether to apply distillation only on completion tokens."},
    )
    distillation_alpha: float = field(
        default=1.0,
        metadata={"help": "KL divergence direction. `1.0` is reverse KL: KL(student || teacher)."},
    )
    distillation_topk: int | None = field(
        default=100,
        metadata={"help": "Number of student top tokens to keep for top-k distillation."},
    )
    distillation_add_tail: bool = field(
        default=True,
        metadata={"help": "Whether to add a tail bucket for non-top-k probability mass."},
    )
    distillation_chunk_size: int | None = field(
        default=1024,
        metadata={
            "help": "Number of completion tokens to process at a time for memory-efficient top-k distillation."
        },
    )
    old_per_token_logps_column: str | None = field(
        default=None,
        metadata={"help": "Optional dataset column containing old per-token log probabilities."},
    )

    def __post_init__(self):
        SelfDistillationConfig.__post_init__(self)

        if self.num_loss_tokens_to_skip < 0:
            raise ValueError("num_loss_tokens_to_skip must be non-negative")
        if not self.loss_on_completion_only:
            raise ValueError("OfflineSDFTTrainer only supports `loss_on_completion_only=True`.")
        if self.sync_ref_model:
            raise ValueError("OfflineSDFTTrainer does not support `sync_ref_model=True`.")
        if self.distillation_chunk_size is not None and self.distillation_chunk_size <= 0:
            raise ValueError("distillation_chunk_size must be positive when provided")
        required_template_fields = ["{prompt}", "{feedback}", "{correct_solution}"]
        if any(field_name not in self.teacher_prompt_template for field_name in required_template_fields):
            raise ValueError(
                "teacher_prompt_template must contain `{prompt}`, `{feedback}`, and `{correct_solution}` placeholders"
            )
