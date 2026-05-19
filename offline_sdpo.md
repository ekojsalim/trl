# Offline Static-Rollout Self-Distillation Trainer Plan

## Decision Summary

Build a new SDFT/SDPO-adjacent trainer in TRL for offline/static rollout self-distillation.

Recommended name:

```text
OfflineSDFTTrainer
OfflineSDFTConfig
```

Recommended location:

```text
trl/experimental/sdft/offline_sdft_trainer.py
trl/experimental/sdft/offline_sdft_config.py
```

This should not be implemented as a `DPOTrainer` modification. The training signal is not "chosen completion beats rejected completion." It is a distributional self-distillation signal along a fixed rollout trajectory.

This should also not be implemented by heavily modifying `SDPOTrainer`. Existing SDPO is built around online generation, reward functions, generation groups, and successful-rollout mining. The offline/static use case removes that whole path.

The closest existing trainer is `SDFTTrainer`: it already performs same-model self-distillation with a privileged teacher prompt. The main missing piece is that SDFT currently generates completions online, while this project needs to consume static failed completions from the dataset.

The cleanest trainer contract should be deliberately small:

```python
{
    "student_prompt": str | list[dict],
    "teacher_prompt": str | list[dict],
    "completion": str | list[dict],
}
```

The experiment-specific fields `prompt`, `feedback`, and `correct_solution` are one way to produce `student_prompt` and `teacher_prompt`. They should be supported through preprocessing or a helper formatter, not treated as the core trainer contract.

## Target Experiment

The target configuration is:

```text
failed_rollout + judge_feedback + correct_solution
```

For each training row, train on the failed rollout trajectory while giving the self-teacher privileged context containing both:

- judge feedback for the failed rollout
- a verified correct on-policy solution for the same prompt/group

The failed completion is not a positive target. It is the trajectory used to define prefixes. At each completion prefix, compare:

```text
student: πθ(. | prompt, failed_completion_<t)
teacher: stopgrad(πθ(. | teacher_prompt(prompt, feedback, correct_solution), failed_completion_<t))
```

Gradients flow only through the student distribution.

## Existing TRL Findings

TRL already has the right core pieces:

- `trl/experimental/sdft/SDFTTrainer` builds student and teacher prompts from `prompt` plus privileged context.
- `trl/experimental/sdpo/SDPOTrainer` shows how successful rollouts and feedback are formatted into teacher reprompts.
- `trl/experimental/self_distillation/SelfDistillationMixin` owns the reusable batch contract and self-distillation loss.
- `trl/experimental/self_distillation/teacher_context.py` contains shared prompt tokenization and chat-template handling.

The existing self-distillation batch contract is the right interface:

```python
{
    "prompt_ids": ...,                 # student prompt ids
    "prompt_mask": ...,                # student prompt attention mask
    "completion_ids": ...,             # static failed completion ids
    "completion_mask": ...,            # loss mask over completion tokens
    "teacher_input_ids": ...,          # teacher prompt + same completion
    "teacher_attention_mask": ...,     # teacher prompt mask + same completion mask
}
```

`SelfDistillationMixin._compute_self_distillation_loss()` already:

- concatenates `prompt_ids + completion_ids` for the student forward pass
- uses `teacher_input_ids` for the teacher forward pass
- slices the shifted logits down to the last `completion_ids.size(1)` positions
- applies the loss only where `completion_mask` is active
- runs the teacher forward pass under `torch.no_grad()`
- supports full-logit, top-k, and optional top-k tail-bucket distillation

This means the first implementation should mostly be a static batch builder plus a thin trainer wrapper.

## Dataset Contract

The core trainer should care only about the three pieces required by the objective:

Required columns:

```python
{
    "student_prompt": str | list[dict],
    "teacher_prompt": str | list[dict],
    "completion": str | list[dict],
}
```

This is the most defensible minimal API because the self-distillation loss does not care where the teacher context came from. It may come from judge feedback, a correct solution, a rubric, hidden traces, synthetic critique, or a fully pre-rendered prompt.

For this project, provide a convenience preprocessing path from the experiment-specific schema:

```python
{
    "prompt": str | list[dict],
    "completion": str | list[dict],
    "feedback": str,
    "correct_solution": str | list[dict],
}
```

into:

```python
{
    "student_prompt": prompt,
    "teacher_prompt": render_teacher_prompt(prompt, feedback, correct_solution),
    "completion": completion,
}
```

Optional metadata columns can be retained and passed through:

```python
{
    "reward": float,
    "group_id": str,
    "old_per_token_logps": list[float],
    "generator_checkpoint": str,
    "judge_version": str,
}
```

For chat datasets:

- `student_prompt` should be a conversation ending in a user turn.
- `teacher_prompt` should also be a prompt-like conversation ending in a user turn, usually with the same system messages as `student_prompt`.
- `completion` should represent the assistant continuation being scored.
- In the convenience raw schema, `correct_solution` should represent an assistant answer or text that can be inserted into the teacher prompt.

The first implementation can require one row per scored rollout. Grouped rollout selection can be added later as preprocessing that chooses a `correct_solution` and renders `teacher_prompt`.

## Teacher Prompt Construction

The trainer should accept pre-rendered `teacher_prompt` rows directly.

For the target experiment, include a helper or dataset preprocessing function with this default template:

```text
{prompt}

Correct solution:
{correct_solution}

The following is judge feedback from an earlier unsuccessful attempt:
{feedback}

Correctly solve the original problem.
```

For string prompts, render this directly into `teacher_prompt`.

For chat prompts, preserve earlier system messages from `prompt` and replace the final user message with the rendered teacher text. This matches the pattern already used by SDFT/SDPO teacher prompt builders.

The original failed `completion` is then appended as the assistant continuation to be scored teacher-forced. It should not be inserted into the teacher prompt body.

## Batch Construction

For each row:

1. Read `student_prompt`, `teacher_prompt`, and `completion`.
2. Tokenize the student prompt with TRL's existing prompt/chat-template helper.
3. Tokenize the teacher prompt with the same prompt/chat-template helper.
4. Tokenize the static failed completion.
5. Right-pad completions and left-pad prompts.
6. Build:

```python
student_input_ids = concat(prompt_ids, completion_ids)
student_attention_mask = concat(prompt_mask, completion_mask)

teacher_input_ids = concat(teacher_prompt_ids, completion_ids)
teacher_attention_mask = concat(teacher_prompt_mask, completion_mask)
```

7. Return the existing `SelfDistillationMixin` batch contract.

Completion alignment does not require student and teacher prompts to have equal length. The shared loss slices the shifted logits to the last `C = completion_ids.size(1)` positions for both student and teacher. Since the same `completion_ids` are appended to both contexts, those `C` positions correspond to the same target completion tokens.

## Loss Semantics

Use SDPO-style top-K + tail logit distillation as the default objective.

At each completion prefix:

1. Compute student next-token logits.
2. Select top-K token ids from the student logits.
3. Compute teacher logits under `torch.no_grad()`.
4. Gather teacher probabilities on the same student top-K token ids.
5. Add one tail bucket for remaining probability mass for both student and teacher.
6. Compute `KL(student || stopgrad(teacher))` over the resulting `K + 1` distribution.

Recommended defaults:

```python
distillation_alpha = 1.0
distillation_topk = 100
distillation_add_tail = True
temperature = 1.0
```

In TRL's existing self-distillation convention:

- `distillation_alpha = 0.0` is forward KL
- `distillation_alpha = 0.5` is JSD
- `distillation_alpha = 1.0` is reverse KL, which corresponds to `KL(student || teacher)`

JSD can remain an explicit experiment, but it should not be the default if the goal is to match the SDPO paper's top-K + tail objective.

Important memory nuance: TRL's existing top-K path appears objective-compatible, but it may still materialize full `[batch, completion, vocab]` student and teacher logits before reducing to top-K. That is acceptable for Phase 1 correctness. It does not fully realize the paper's peak-memory-efficiency claim.

## Minimal API

Prefer extending existing TRL self-distillation config names instead of introducing parallel aliases.

```python
class OfflineSDFTConfig(SDFTConfig):
    student_prompt_column: str = "student_prompt"
    teacher_prompt_column: str = "teacher_prompt"
    completion_column: str = "completion"

    # Optional convenience formatter for raw experiment rows. If an example already
    # has teacher_prompt_column, the trainer should use it directly and ignore these fields.
    prompt_column: str = "prompt"
    feedback_column: str = "feedback"
    correct_solution_column: str = "correct_solution"
    teacher_prompt_template: str = (
        "{prompt}\n\n"
        "Correct solution:\n{correct_solution}\n\n"
        "The following is judge feedback from an earlier unsuccessful attempt:\n"
        "{feedback}\n\n"
        "Correctly solve the original problem."
    )

    append_eos_token: bool = True
    loss_on_completion_only: bool = True

    distillation_alpha: float = 1.0
    distillation_topk: int | None = 100
    distillation_add_tail: bool = True
    temperature: float = 1.0

    old_per_token_logps_column: str | None = None
```

Phase 1 can be even simpler: require `student_prompt`, `teacher_prompt`, and `completion`, and provide a standalone preprocessing helper for the raw `prompt`/`feedback`/`correct_solution` schema. Configurable raw-column support can be added after the core trainer works.

## Implementation Plan

### Phase 1: Correct Static Trainer

Implement `OfflineSDFTTrainer` by reusing the SDFT initialization and `SelfDistillationMixin`.

Primary changes:

- Add a static completion tokenizer.
- Replace SDFT's online `_generate_completion_ids()` path with dataset completion tokenization.
- Accept pre-rendered `student_prompt` and `teacher_prompt` rows.
- Add an optional helper to render `teacher_prompt` from `prompt`, `feedback`, and `correct_solution`.
- Return the existing self-distillation batch contract.
- Keep teacher `no_grad` and PEFT/EMA behavior from SDFT.

Files to add or modify:

```text
trl/experimental/sdft/offline_sdft_config.py
trl/experimental/sdft/offline_sdft_trainer.py
trl/experimental/sdft/__init__.py
docs/source/offline_sdft_trainer.md
tests/experimental/test_offline_sdft_trainer.py
```

Optional script after the trainer lands:

```text
trl/experimental/sdft/offline_sdft.py
```

### Phase 2: Paper-Like Defaults and Tests

Set defaults to:

```python
distillation_alpha = 1.0
distillation_topk = 100
distillation_add_tail = True
```

Add tests that verify:

- the static failed completion is used, not generated
- student and teacher inputs share identical completion ids
- completion masks have the expected length
- teacher completion attention equals `completion_mask`
- loss is finite
- top-K + tail path is active
- gradients flow through the student path
- teacher logits are stop-gradient/no-grad

Suggested synthetic row:

```python
{
    "student_prompt": "Return the numbers from 1 to n, excluding n.",
    "teacher_prompt": (
        "Return the numbers from 1 to n, excluding n.\n\n"
        "Correct solution:\n"
        "def f(n): return list(range(1, n))\n\n"
        "The following is judge feedback from an earlier unsuccessful attempt:\n"
        "The output incorrectly includes n.\n\n"
        "Correctly solve the original problem."
    ),
    "completion": "def f(n): return list(range(1, n + 1))",
}
```

Also test the convenience preprocessing shape:

```python
{
    "prompt": "Return the numbers from 1 to n, excluding n.",
    "completion": "def f(n): return list(range(1, n + 1))",
    "feedback": "The output incorrectly includes n.",
    "correct_solution": "def f(n): return list(range(1, n))",
}
```

Also add a chat-format test where:

```python
prompt = [{"role": "user", "content": "..."}]
completion = [{"role": "assistant", "content": "..."}]
```

### Phase 3: Peak-Memory Optimization

Only do this if memory becomes the limiting factor.

Target behavior:

- compute student logits for completion positions in chunks
- select student top-K ids per chunk
- compute teacher logits under `torch.no_grad()`
- immediately gather teacher probability mass on the student top-K ids
- compute tail mass
- discard full teacher logits before processing the next chunk

This would make the implementation closer to the SDPO paper's memory-efficient Appendix A.3 objective. It is not required for the first correctness implementation.

## Pseudocode

```python
class OfflineSDFTTrainer(SelfDistillationMixin, _BaseTrainer):
    config_cls = OfflineSDFTConfig

    def _prepare_inputs(self, examples):
        student_prompts = [ex[self.args.student_prompt_column] for ex in examples]
        teacher_prompts = [ex[self.args.teacher_prompt_column] for ex in examples]
        completions = [ex[self.args.completion_column] for ex in examples]

        student_batch = self.prompt_tokenizer.tokenize_prompts(student_prompts)
        teacher_batch = self.prompt_tokenizer.tokenize_prompts(teacher_prompts)

        completion_ids, completion_mask = self._tokenize_static_completions(student_prompts, completions)

        return {
            "prompt_ids": student_batch.prompt_ids,
            "prompt_mask": student_batch.prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_input_ids": torch.cat([teacher_batch.prompt_ids, completion_ids], dim=1),
            "teacher_attention_mask": torch.cat([teacher_batch.prompt_mask, completion_mask], dim=1),
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("OfflineSDFTTrainer does not support returning outputs")
        return self._compute_self_distillation_loss(model, inputs) / self.current_gradient_accumulation_steps
```

Completion tokenization should be the most carefully tested piece. For chat data, use the tokenizer's chat template to derive the completion suffix from `student_prompt + completion` so assistant-turn markers and EOS behavior match the model's training format.

If supporting raw experiment rows directly inside the trainer, add a small normalization step before tokenization:

```python
if "teacher_prompt" not in row:
    row["student_prompt"] = row["prompt"]
    row["teacher_prompt"] = render_teacher_prompt(row["prompt"], row["feedback"], row["correct_solution"])
```

## Risks and Gotchas

### Completion Is Not a Supervised Target

The failed completion is teacher-forced only to define prefixes. Do not add CE/NLL on the failed text.

### Student and Teacher Prompt Lengths Differ

This is expected. Alignment is by the final `completion_ids.size(1)` shifted logits, not by absolute sequence position.

### Chat Templates Can Duplicate Roles

For chat data, avoid putting the failed completion into the teacher prompt text. It must remain the appended assistant continuation.

### EOS Handling Must Be Explicit

Decide whether static completions should append EOS by default. The recommended default is `append_eos_token=True`, but it should be configurable and covered by tests.

### Packed Sequences Are Out of Scope Initially

Do not support packing in Phase 1. The prompt/teacher prompt lengths differ per row, so packing would complicate alignment and masks.

### Off-Policy Drift

The rollouts were on-policy at collection time but become off-policy during training. Phase 1 can rely on conservative LR and early stopping. Future support for `old_per_token_logps` can add clipped importance correction.

### PEFT/LoRA

Reuse SDFT's PEFT behavior. For PEFT models, the teacher path can use the base model with adapters disabled or an EMA teacher adapter when enabled.

## Acceptance Criteria

The first implementation is acceptable when:

- a tiny static dataset trains without online generation or reward functions
- string and chat prompt/completion examples tokenize correctly
- student and teacher completion ids match exactly
- loss is applied only to completion tokens
- teacher forward pass is no-grad
- loss is finite with top-K + tail KL defaults
- a tiny overfit run reduces the distillation loss
- existing SDFT/SDPO tests still pass
