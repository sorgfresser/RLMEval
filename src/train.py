from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from rlm_eval.metrics.beq_plus import check_theorem_eq_server, UVICORN_PORT, try_repeatedly
from rlm_eval.data_processing.lean_utils import trim_comments_end
import argparse
import requests
from lean_pool_models import RunRequest
from lean_interact import Command
from lean_interact.interface import CommandResponse, LeanError
from pydantic import ValidationError
import concurrent.futures
import logging
import os
from transformers import AutoTokenizer
from FlagEmbedding import BGEM3FlagModel
import torch
import numpy

torch.serialization.add_safe_globals(
    [numpy.core.multiarray._reconstruct, numpy.ndarray, numpy.dtype, numpy.dtypes.UInt32DType])

logger = logging.getLogger(__name__)

MAX_WORKERS = int(os.getenv("LEAN_POOL_MAX_WORKERS", 1))
model_path = "purewhite42/dependency_retriever_f"
model = BGEM3FlagModel(model_path, use_fp16=True)
batch_size = 64


def format_doc_only_f(decl: str) -> str:
    return f'''Formal Declaration: {decl[:1536]}'''


def get_embedding_sim(completions: list[list[dict[str, str]]], informal: list[str], **kwargs) -> list[float]:
    completions = [completion[0]["content"] for completion in completions]
    formal_statements = [format_doc_only_f(pred) for pred in completions]
    informal_statements = informal

    formal_embeddings = model.encode(
        formal_statements,
        batch_size=batch_size,
        max_length=1024)['dense_vecs']
    informal_embeddings = model.encode(
        informal_statements,
        batch_size=batch_size,
        max_length=1024
    )['dense_vecs']

    similarity = (torch.tensor(informal_embeddings).double() @ torch.tensor(formal_embeddings).double().T)

    diagonals = similarity.diagonal()
    return diagonals.tolist()


def get_beqplus(context: str, ground_truth: str, prediction: str, project: str, is_theorem: bool,
                beqplus: bool = True) -> float:
    # check if the last line ends with " in" and remove " in" if it does
    trimmed_context = trim_comments_end(context).rstrip()
    if "\n" in trimmed_context:
        lean_context, last_line = trimmed_context.rsplit("\n", 1)
    else:
        lean_context = ""
        last_line = trimmed_context
    if last_line.endswith(" in"):
        lean_context += "\n" + last_line[:-3]
    else:
        lean_context = context
    if not lean_context.strip():
        lean_context = "-- context stub"

    req = RunRequest(data=Command(cmd=ground_truth), context=lean_context)
    try:
        try_repeatedly(req, project)
    except (ValidationError, RuntimeError):
        logger.exception("Failed to validate ground-truth!")
        raise

    if not prediction:
        well_typed, beq_result = False, None
    else:
        req = RunRequest(data=Command(cmd=prediction), context=lean_context)
        prediction_response = try_repeatedly(req, project, allow_error=True)
        try:
            prediction_data = CommandResponse.model_validate(prediction_response.result)
        except ValidationError:
            prediction_data = LeanError.model_validate(prediction_response.result)
        if isinstance(prediction_data, LeanError):
            well_typed, beq_result = False, None
        else:
            well_typed = prediction_data.lean_code_is_valid(allow_sorry=is_theorem)
            beq_result = None
            if well_typed and is_theorem and beqplus:
                beq_result = check_theorem_eq_server(
                    theorem1=prediction,
                    theorem2=ground_truth,
                    context=context,
                    project=project,
                )

    reward = 0.0
    if well_typed:
        reward += 1.0
        if beq_result and beq_result.beql():
            reward += 1.0
        if beq_result and beq_result.beq_plus():
            reward += 1.0
    return reward


def make_conversational(batch):
    all_messages = []
    for name, problem, context in zip(batch["name"], batch["informal"], batch["context"], strict=True):
        prompt = f"Please autoformalize the following problem in Lean 4 with a header. Here is some context for the problem:\n\n{context}\n\nNow formalize the problem. Use the following theorem names: {name}.\n\n"
        prompt += problem

        messages = [
            {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
            {"role": "user", "content": prompt}
        ]
        all_messages.append(messages)
    batch["prompt"] = all_messages
    return batch


def split_func(sample, train: bool = True):
    if train:
        return sample["project"] != "FLT"
    return sample["project"] == "FLT"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--server-ip", type=str, default="localhost")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--lean-weight", type=float, default=1.0, help="Weight for the Lean BEQ+ reward. Only useful if --embedding-sim is set, otherwise simply a multiplier for the BEQ+ reward.")
    parser.add_argument("--embedding-sim", action="store_true", default=False)
    parser.add_argument("--no-beq", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    dataset = load_dataset("sorgfresser/autoformtest", split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def filter_length(elem, length: int = tokenizer.model_max_length):
        if len(tokenizer(tokenizer.apply_chat_template(elem["prompt"], tokenize=False,
                                                       add_generation_prompt=True)).input_ids) > length:
            return False
        return True

    # Rename informal to prompt, at some point one might want to add context
    dataset = dataset.map(make_conversational, batch_size=1000, batched=True)
    dataset = dataset.filter(lambda x: filter_length(x, length=tokenizer.model_max_length - 1024))
    train_dataset = dataset.filter(split_func)
    test_dataset = dataset.filter(lambda x: not split_func(x))

    def reward_num_unique_chars(completions, context, ground_truth, project, is_theorem, **kwargs):
        # If multiple output messages, immediately bs
        rewards = [None if len(c) == 1 else 0.0 for c in completions]
        # If role is wrong, immediately bs
        rewards = [r if c and c[0]["role"] == "assistant" else 0.0 for r, c in zip(rewards, completions, strict=True)]
        completion_strings = [None if r is not None else c[0]["content"] for r, c in
                              zip(rewards, completions, strict=True)]
        beqplus = not args.no_beq
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(get_beqplus, con, g, c, p, t, beqplus): idx for idx, (con, g, c, p, t) in
                       enumerate(zip(context, ground_truth, completion_strings, project, is_theorem, strict=True))}
            failures = 0
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    rewards[idx] = fut.result()
                except Exception:
                    logger.info("Reward thread %s failed", idx)
                    failures += 1
                    rewards[idx] = None  # failed, so cannot use this reward
            logger.warning("%s of %s threads failed", failures, len(futures))
            if failures > len(futures) // 5:
                logger.error("More than a fifth of all reward threads failed!")
        # Only reset on master and only once everything is processed
        trainer.accelerator.wait_for_everyone()  # sync them, cause master might otherwise reset too early
        if training_args.process_index == 0:
            resp = requests.post(f"http://localhost:{UVICORN_PORT}/reset")
            resp.raise_for_status()
        return rewards

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=3,
        weight_decay=0.1,
        use_vllm=True,
        num_generations=48,
        warmup_steps=5,
        report_to=["wandb"],
        max_completion_length=1024,
        beta=args.beta,
        vllm_server_host=args.server_ip,
        loss_type="dr_grpo",
        learning_rate=1e-6,
        eval_strategy="steps",
        eval_steps=51,
        per_device_eval_batch_size=16,
        save_strategy="steps",
        save_steps=25,
        push_to_hub=True,
        lr_scheduler_type="constant_with_warmup",
        reward_weights=[args.lean_weight, 1.0] if args.embedding_sim else [args.lean_weight],
    )
    reward_funcs = [reward_num_unique_chars]
    if args.embedding_sim:
        reward_funcs.append(get_embedding_sim)

    trainer = GRPOTrainer(model=args.model_name, args=training_args,
                          reward_funcs=reward_funcs, train_dataset=train_dataset,
                          eval_dataset=test_dataset)
    trainer.train(resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
