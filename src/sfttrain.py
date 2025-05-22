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

MAX_WORKERS = int(os.getenv("LEAN_POOL_MAX_WORKERS", 1))


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
    for name, problem, context in zip(batch["name"], batch["prompt"], batch["context"], strict=True):
        prompt = f"Please autoformalize the following problem in Lean 4 with a header. Here is some context for the problem:\n\n{context}\n\nNow formalize the problem. Use the following theorem names: {name}.\n\n"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", type=str, default="localhost")
    args = parser.parse_args()
    from datasets import load_dataset
    from rlm_eval.metrics.beq_plus import check_theorem_eq_server
    from rlm_eval.data_processing.lean_utils import trim_comments_end

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
        resp = requests.post("http://localhost:8080/reset")
        resp.raise_for_status()
        return rewards
    #
    # training_args = GRPOConfig(
    #     output_dir="Kimina-GRPO",
    #     per_device_train_batch_size=2,
    #     gradient_accumulation_steps=64,
    #     bf16=True,
    #     gradient_checkpointing=True,
    #     logging_steps=10,
    #     weight_decay=0.1,
    #     use_vllm=True,
    #     num_generations=16,
    #     warmup_steps=5,
    #     report_to=["wandb"],
    #     max_completion_length=1024,
    #     beta=0,
    #     vllm_server_host=args.server_ip,
    #     loss_type="dr_grpo",
    #     learning_rate=1e-6
    # )

    # trainer = GRPOTrainer(model="Qwen/Qwen2.5-Coder-0.5B-Instruct", args=training_args,
    #                       reward_funcs=[reward_num_unique_chars], train_dataset=dataset, )
    # trainer.train()
    def split_func(sample, train: bool = True):
        if train:
            return sample["project"] != "FLT"
        return sample["project"] == "FLT"


    from transformers import AutoConfig, AutoTokenizer, Qwen2ForCausalLM, EvalPrediction
    from trl import SFTConfig, SFTTrainer
    import evaluate
    dataset = dataset.rename_columns({"ground_truth": "completion"})
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
    dataset = dataset.filter(lambda x: filter_length(x, tokenizer.model_max_length // 64))
    train_dataset = dataset.filter(split_func)
    test_dataset = dataset.filter(lambda x: not split_func(x))
    test_dataset = test_dataset.filter(lambda x: filter_length(x, tokenizer.model_max_length // 32))
    training_args = SFTConfig(
        max_length=tokenizer.model_max_length,
        output_dir="./testtrainsft",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=2,
        eval_steps=10,
        eval_strategy="steps",
        eval_on_start=False,
        num_train_epochs=1,
        padding_free=True,
        eval_accumulation_steps=1
    )

    trainer = SFTTrainer(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("./testtrainsft")
    # model = LLM("AI-MO/Kimina-Autoformalizer-7B")
    # tokenizer = AutoTokenizer.from_pretrained("AI-MO/Kimina-Autoformalizer-7B")
    # inp = next(iter(dataset))

    # prompt = inp["prompt"]
    # text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    # sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048)
    # output = model.generate(text, sampling_params=sampling_params)
    # output_text = output[0].outputs[0].text
    # print(output_text)


