from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from rlm_eval.metrics.beq_plus import check_theorem_eq_server
from rlm_eval.data_processing.lean_utils import trim_comments_end
import argparse
import requests
from lean_pool_models import ContextRequest, ContextResponse, RunResponse, RunRequest
from lean_interact import Command
from lean_interact.interface import CommandResponse, LeanError
from pydantic import ValidationError
import concurrent.futures
import logging
import os

logger = logging.getLogger(__name__)



def make_conversational(batch):
    all_messages = []
    for problem, context in zip(batch["prompt"], batch["context"], strict=True):
        prompt = f"Please autoformalize the following problem in Lean 4 with a header. Here is some context for the problem:\n\n{context}\n\nNow formalize the problem. Use the following theorem names: thm_P.\n\n"
        # prompt = f"Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: {name}.\n\n"
        prompt += problem

        messages = [
            {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
            {"role": "user", "content": prompt}
        ]
        all_messages.append(messages)
    batch["prompt"] = all_messages
    return batch


def main():
    dataset = load_dataset("sorgfresser/connf_with_context", split="train")

    # Rename informal to prompt, at some point one might want to add context
    dataset = dataset.rename_columns({"informal": "prompt"})
    dataset = dataset.map(make_conversational, batch_size=1000, batched=True)

    from transformers import AutoConfig, AutoTokenizer, Qwen2ForCausalLM, EvalPrediction
    from trl import SFTConfig, SFTTrainer
    import evaluate
    dataset = dataset.rename_columns({"formal": "completion"})
    tokenizer = AutoTokenizer.from_pretrained("AI-MO/Kimina-Autoformalizer-7B")
    bleu = evaluate.load("bleu")
    def to_chat_template(batch):
        prompts = []
        for prompt in batch["prompt"]:
            prompts.append(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True))
        batch["prompt"] = prompts
        return batch
    def filter_length(elem, length: int = tokenizer.model_max_length):
        if len(tokenizer(elem["prompt"] + elem["completion"]).input_ids) > length:
            return False
        return True
    def compute_metrics(pred: EvalPrediction):
        max_preds = pred.predictions.argmax(-1)
        mask = pred.label_ids != -100
        acc = (pred.label_ids == max_preds)[mask].mean()
        labels = [string + tokenizer.eos_token for string in tokenizer.decode(pred.label_ids[mask]).split(tokenizer.eos_token)[:-1]]
        for idx, b in enumerate(mask):
            if not max_preds[idx, b][-1] == tokenizer.eos_token_id:
                max_preds[idx, b][-1] = tokenizer.eos_token_id
        preds = [string + tokenizer.eos_token for string in tokenizer.decode(max_preds[mask]).split(tokenizer.eos_token)[:-1]]
        preds = preds if len(preds) <= len(labels) else preds[:len(labels)]
        return {"accuracy": acc, "bleu": bleu.compute(predictions=preds, references=labels)["bleu"]}

    dataset = dataset.map(to_chat_template, batched=True, batch_size=1000)
    dataset = dataset.filter(filter_length)
    training_args = SFTConfig(
        max_length=tokenizer.model_max_length,
        output_dir="./testtrainsft",
        bf16=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=2,
        eval_on_start=False,
        num_train_epochs=4,
        gradient_checkpointing=True,
        eval_accumulation_steps=1,
        learning_rate=7e-6,
    )

    trainer = SFTTrainer(
        "AI-MO/Kimina-Autoformalizer-7B",
        train_dataset=dataset,
        args=training_args,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("./testtrainsft")

if __name__ == "__main__":
    main()
