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

import torch
from datasets import Dataset

from trl.experimental.sdft import OfflineSDFTConfig, OfflineSDFTTrainer, render_offline_sdft_teacher_prompt

from ..testing_utils import TrlTestCase


class TestOfflineSDFTTrainer(TrlTestCase):
    def test_config_defaults_use_topk_tail_reverse_kl(self):
        config = OfflineSDFTConfig(output_dir=self.tmp_dir)

        assert config.num_generations == 1
        assert config.distillation_alpha == 1.0
        assert config.distillation_topk == 100
        assert config.distillation_add_tail is True
        assert config.distillation_chunk_size == 16
        assert config.distillation_chunk_backend == "hidden_state"
        assert config.append_eos_token is True

    def test_render_offline_teacher_prompt_for_chat_preserves_prefix_messages(self):
        prompt = [
            {"role": "system", "content": "Use concise Python."},
            {"role": "user", "content": "Return the numbers from 1 to n, excluding n."},
        ]

        teacher_prompt = render_offline_sdft_teacher_prompt(
            prompt=prompt,
            feedback="The output incorrectly includes n.",
            correct_solution="def f(n): return list(range(1, n))",
        )

        assert teacher_prompt[0] == prompt[0]
        assert teacher_prompt[-1]["role"] == "user"
        assert "Correct solution:" in teacher_prompt[-1]["content"]
        assert "def f(n): return list(range(1, n))" in teacher_prompt[-1]["content"]
        assert "The output incorrectly includes n." in teacher_prompt[-1]["content"]

    def test_prepare_inputs_uses_static_completion_without_generation(self):
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
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=32,
            distillation_topk=5,
            append_eos_token=False,
        )
        trainer = OfflineSDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        def fail_generation(*args, **kwargs):
            raise AssertionError("OfflineSDFTTrainer should not generate completions")

        trainer._generate_completion_ids = fail_generation
        batch = trainer._prepare_inputs([dataset[0]])
        completion_width = batch["completion_ids"].size(1)

        assert completion_width == batch["completion_mask"].sum().item()
        assert torch.equal(batch["teacher_input_ids"][:, -completion_width:], batch["completion_ids"])
        assert torch.equal(batch["teacher_attention_mask"][:, -completion_width:], batch["completion_mask"])
        decoded_completion = trainer.processing_class.decode(batch["completion_ids"][0], skip_special_tokens=True)
        assert "n + 1" in decoded_completion

    def test_prepare_inputs_supports_raw_feedback_schema(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Return the numbers from 1 to n, excluding n."],
                "completion": ["def f(n): return list(range(1, n + 1))"],
                "feedback": ["The output incorrectly includes n."],
                "correct_solution": ["def f(n): return list(range(1, n))"],
            }
        )
        training_args = OfflineSDFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=32,
            distillation_topk=5,
            append_eos_token=False,
        )
        trainer = OfflineSDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        batch = trainer._prepare_inputs([dataset[0]])
        decoded_teacher = trainer.processing_class.decode(batch["teacher_input_ids"][0], skip_special_tokens=True)

        assert "Correct solution:" in decoded_teacher
        assert "The output incorrectly includes n." in decoded_teacher
        assert "n + 1" in decoded_teacher

    def test_static_completion_truncates_from_right(self):
        dataset = Dataset.from_dict(
            {
                "student_prompt": ["Continue."],
                "teacher_prompt": ["Continue with privileged context."],
                "completion": [" alpha beta gamma delta epsilon zeta"],
            }
        )
        training_args = OfflineSDFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=3,
            distillation_topk=5,
            append_eos_token=False,
        )
        trainer = OfflineSDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        batch = trainer._prepare_inputs([dataset[0]])
        truncation_side = trainer._tokenizer.truncation_side
        trainer._tokenizer.truncation_side = "right"
        try:
            expected_ids = trainer.processing_class(
                text=[dataset[0]["completion"]],
                return_tensors="pt",
                max_length=3,
                truncation=True,
                add_special_tokens=False,
            )["input_ids"]
        finally:
            trainer._tokenizer.truncation_side = truncation_side

        assert torch.equal(batch["completion_ids"].cpu(), expected_ids)

    def test_compute_loss_and_evaluate_static_completion(self):
        dataset = Dataset.from_dict(
            {
                "student_prompt": ["Return the numbers from 1 to n, excluding n."],
                "teacher_prompt": [
                    "Return the numbers from 1 to n, excluding n.\n\n"
                    "Correct solution:\n"
                    "def f(n): return list(range(1, n))\n\n"
                    "Correctly solve the original problem."
                ],
                "completion": ["def f(n): return list(range(1, n + 1))"],
            }
        )
        training_args = OfflineSDFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            max_steps=1,
            max_completion_length=32,
            distillation_topk=5,
            append_eos_token=False,
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
        )
        trainer = OfflineSDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
        )

        batch = trainer._prepare_inputs([dataset[0]])
        loss = trainer.compute_loss(trainer.model, batch)
        trainer.args.distillation_weight = 0.0
        zero_weight_loss = trainer.compute_loss(trainer.model, batch)
        trainer.args.distillation_weight = 1.0
        metrics = trainer.evaluate()
        train_result = trainer.train()

        assert torch.isfinite(loss)
        assert zero_weight_loss.item() == 0.0
        assert "eval_loss" in metrics
        assert torch.isfinite(torch.tensor(train_result.training_loss))

    def test_prepare_inputs_supports_chat_completion_suffix(self):
        dataset = Dataset.from_dict(
            {
                "student_prompt": [[{"role": "user", "content": "Solve 2+2."}]],
                "teacher_prompt": [
                    [
                        {
                            "role": "user",
                            "content": "Solve 2+2.\n\nCorrect solution:\n4\n\nCorrectly solve the original problem.",
                        }
                    ]
                ],
                "completion": [[{"role": "assistant", "content": "The answer is 5."}]],
            }
        )
        training_args = OfflineSDFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=32,
            distillation_topk=5,
            append_eos_token=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        trainer = OfflineSDFTTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        batch = trainer._prepare_inputs([dataset[0]])
        completion_width = batch["completion_ids"].size(1)

        assert torch.equal(batch["teacher_input_ids"][:, -completion_width:], batch["completion_ids"])
        decoded_completion = trainer.processing_class.decode(batch["completion_ids"][0], skip_special_tokens=True)
        assert "The answer is 5." in decoded_completion
