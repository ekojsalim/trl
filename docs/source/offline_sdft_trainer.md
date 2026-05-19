# Offline SDFT

Offline SDFT is an experimental static-rollout self-distillation trainer for datasets where rollouts have already been
collected.

Unlike [`SDFTTrainer`](sdft_trainer), it does not generate completions during training. Unlike preference trainers, it
does not train a chosen completion against a rejected completion. Instead, each row provides one fixed completion
trajectory, and the trainer distills:

```text
student: model(. | student_prompt, completion_prefix)
teacher: stopgrad(model(. | teacher_prompt, completion_prefix))
```

The same fixed completion is appended to both prompts. Gradients flow only through the student-conditioned
distribution.

## Usage

```python
from datasets import Dataset

from trl.experimental.sdft import OfflineSDFTConfig, OfflineSDFTTrainer

dataset = Dataset.from_dict(
    {
        "student_prompt": ["Return the numbers from 1 to n, excluding n."],
        "teacher_prompt": [
            "Return the numbers from 1 to n, excluding n.\n\n"
            "Correct solution:\n"
            "def f(n): return list(range(1, n))\n\n"
            "The following is judge feedback from an earlier unsuccessful attempt:\n"
            "The output incorrectly includes n.\n\n"
            "Correctly solve the original problem."
        ],
        "completion": ["def f(n): return list(range(1, n + 1))"],
    }
)

training_args = OfflineSDFTConfig(
    output_dir="offline-sdft-model",
    distillation_alpha=1.0,
    distillation_topk=100,
    distillation_add_tail=True,
    max_completion_length=256,
)

trainer = OfflineSDFTTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

## Expected Dataset Columns

The core dataset contract is:

- `student_prompt`: student-facing prompt
- `teacher_prompt`: privileged teacher prompt
- `completion`: fixed rollout completion

The trainer also supports a raw convenience schema. When `teacher_prompt` is absent, it renders one from:

- `prompt`
- `feedback`
- `correct_solution`
- `completion`

For conversational prompts, the raw formatter preserves the earlier prompt messages and replaces the final user message
with the rendered teacher text. The failed rollout remains the appended assistant completion; it is not inserted into
the teacher prompt body.

## Defaults

Offline SDFT uses SDPO-style top-k plus tail-bucket distillation by default:

```python
OfflineSDFTConfig(
    distillation_alpha=1.0,
    distillation_topk=100,
    distillation_add_tail=True,
    distillation_chunk_size=1024,
)
```

Chunked top-k distillation patches the model forward before Accelerate, FSDP, or DeepSpeed wrapping. It runs one
student backbone forward and one teacher backbone forward, then projects hidden states through the LM head in
`distillation_chunk_size` completion-token chunks. The student projection uses a Liger-style custom autograd function,
which avoids saving full-vocab logits and computes the projection gradient directly in backward. The teacher projection
remains no-grad and chunked.

Under FSDP2 with `fsdp_reshard_after_forward=true`, this path gathers and caches the frozen LM head across the training
step so the custom backward does not read a resharded parameter. This currently supports the PEFT-style frozen LM-head
setting. Offline SDFT does not support `sync_ref_model=True`; the teacher is the current model evaluated under the
teacher prompt.

## Logged Metrics

Offline SDFT logs the masked mean `offline_sdft/distillation_loss`, along with distribution diagnostics for the fixed
completion trajectory:

- `offline_sdft/student_topk_mass`
- `offline_sdft/teacher_topk_mass`
- `offline_sdft/student_entropy`
- `offline_sdft/teacher_entropy`

When `old_per_token_logps_column` is provided and importance-sampling clipping is enabled, it also logs:

- `offline_sdft/is_ratio_mean`
- `offline_sdft/is_clipped_frac`

Static completions append the tokenizer EOS token by default after text or chat-template rendering. Set
`append_eos_token=False` if the dataset already contains exactly the token sequence to score.

## OfflineSDFTConfig

[[autodoc]] experimental.sdft.OfflineSDFTConfig

## OfflineSDFTTrainer

[[autodoc]] experimental.sdft.OfflineSDFTTrainer
    - train
    - save_model
    - push_to_hub
