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

MAX_WORKERS = int(os.getenv("LEAN_POOL_MAX_WORKERS", 4))


def get_beqplus(context: str, ground_truth: str, prediction: str, project: str, is_theorem: bool) -> float:
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

    req = ContextRequest(context=lean_context)
    resp = requests.post(f"http://localhost:8080/{project}/context", data=req.model_dump_json())
    response = ContextResponse.model_validate(resp.json())
    env = response.env_id
    server_id = response.server_id
    req = RunRequest(data=Command(cmd=ground_truth, env=env), server_id=server_id)
    resp = requests.post(f"http://localhost:8080/{project}/run", data=req.model_dump_json())
    ground_truth_response = RunResponse.model_validate(resp.json())
    # Check ground truth is valid
    try:
        CommandResponse.model_validate(ground_truth_response.result)
    except ValidationError:
        logger.exception("Failed to validate ground-truth!")
        raise

    if not prediction:
        well_typed, beq_result = False, None
    else:
        req = RunRequest(data=Command(cmd=prediction, env=env), server_id=server_id)
        resp = requests.post(f"http://localhost:8080/{project}/run", data=req.model_dump_json())
        prediction_response = RunResponse.model_validate(resp.json())
        try:
            prediction_data = CommandResponse.model_validate(prediction_response.result)
        except ValidationError:
            prediction_data = LeanError.model_validate(prediction_response.result)
        if isinstance(prediction_data, LeanError):
            well_typed, beq_result = False, None
        else:
            well_typed = prediction_data.lean_code_is_valid(allow_sorry=is_theorem)
            beq_result = None
            if well_typed and is_theorem:
                beq_result = check_theorem_eq_server(
                    theorem1=prediction,
                    theorem2=ground_truth,
                    context_env=env,
                    server_id=server_id,
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
    for name, problem in zip(batch["name"], batch["prompt"], strict=True):
        prompt = f"Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: {name}.\n\n"
        prompt += problem

        messages = [
            {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
            {"role": "user", "content": prompt}
        ]
        all_messages.append(messages)
    batch["prompt"] = all_messages
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", type=str, default="localhost")
    args = parser.parse_args()

    dataset = load_dataset("sorgfresser/autoformtest", split="train")

    # Rename informal to prompt, at some point one might want to add context
    dataset = dataset.rename_columns({"informal": "prompt"})
    dataset = dataset.map(make_conversational, batch_size=1000, batched=True)

    # Dummy reward function: count the number of unique characters in the completions
    def reward_num_unique_chars(completions, context, ground_truth, project, is_theorem, **kwargs):
        # If multiple output messages, immediately bs
        rewards = [None if len(c) == 1 else 0.0 for c in completions]
        # If role is wrong, immediately bs
        rewards = [r if c and c[0]["role"] == "assistant" else 0.0 for r, c in zip(rewards, completions, strict=True)]
        completion_strings = [None if r is not None else c[0]["content"] for r, c in
                              zip(rewards, completions, strict=True)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(get_beqplus, con, g, c, p, t): idx for idx, (con, g, c, p, t) in
                       enumerate(zip(context, ground_truth, completion_strings, project, is_theorem, strict=True))}
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    rewards[idx] = fut.result()
                except Exception:
                    logger.exception("Reward thread %s failed", idx)
                    rewards[idx] = 0.0
        return rewards

    training_args = GRPOConfig(
        output_dir="Kimina-GRPO",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        weight_decay=0.1,
        use_vllm=True,
        num_generations=16,
        warmup_steps=1000,
        report_to=["wandb"],
        max_completion_length=1024,
        beta=0,
        vllm_server_host=args.server_ip,
        loss_type="dr_grpo"
    )

    trainer = GRPOTrainer(model="Qwen/Qwen2.5-Coder-0.5B-Instruct", args=training_args,
                          reward_funcs=[reward_num_unique_chars], train_dataset=dataset, )
    trainer.train()


if __name__ == "__main__":
    main()
