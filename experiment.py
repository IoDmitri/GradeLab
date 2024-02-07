import argparse
import json

from client import client_from_args
from llm_eval import Evaluator
from prompts import judge_prompt, judge_prompt_v2, judge_prompt_v3

from datasets import load_dataset, load_from_disk


def stats_for_dataset(dataset: str, prompt: str, client: str, prompt_key:str, outputs_key: str, is_local=False,
                      llm_bias_threshold=0.4,choice_consistency_threshold=0.6, negative_sampling=False, api_key=None,
                      temperature=0.3, model=None, url=None):

    ds = load_from_disk(dataset) if is_local else load_dataset(dataset)
    if "train" in ds:
        ds = ds["train"]
    print(f"loaded dataset {dataset}")
    if client not in ["openai", "mistral"]:
        raise ValueError(f"Expected 'openai' or 'mistral', got {client}, which is currently unsupported")

    client = client_from_args(client, api_key=api_key, model=model, url=url)

    evaluator = Evaluator(client, prompt)
    gen_args = {"temperature": temperature}
    if model:
        gen_args["model"] = model

    judge_prompt_to_use = judge_prompt
    if prompt in ["2", "prompt_2"]:
        judge_prompt_to_use = judge_prompt_v2
    elif prompt in ["3", "prompt_3"]:
        judge_prompt_to_use = judge_prompt_v3
    elif prompt:
        judge_prompt_to_use = prompt

    return evaluator.grade_stats_for_dataset(
        ds, prompt_key, outputs_key, judge_prompt_to_use, negative_sample=negative_sampling,
        llm_bias_threshold=llm_bias_threshold, choice_consistency_threshold=choice_consistency_threshold,
        **gen_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate dataset with LLMs.")
    parser.add_argument("dataset", type=str, help="Name of the dataset or path if local.")
    parser.add_argument("prompt", type=str, help="Prompt type or custom prompt.")
    parser.add_argument("client", type=str, choices=["openai", "mistral"], help="Client to use: 'openai' or 'mistral'.")
    parser.add_argument("prompt_key", type=str, help="Key for the prompt in the dataset.")
    parser.add_argument("outputs_key", type=str, help="Key for the outputs in the dataset.")
    parser.add_argument("--is_local", action="store_true", help="Flag to load dataset from disk.")
    parser.add_argument("--llm_bias_threshold", type=float, default=0.4, help="LLM bias threshold.")
    parser.add_argument("--choice_consistency_threshold", type=float, default=0.6, help="Choice consistency threshold.")
    parser.add_argument("--negative_sampling", action="store_true", help="Flag for negative sampling.")
    parser.add_argument("--api_key", type=str, help="API key for the client.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for generation.")
    parser.add_argument("--model", type=str, help="Model to be used for generation.")
    parser.add_argument("--save_path", type=str, default="output.json", help="Path to save the output.")
    parser.add_argument("--url", type=str, help="the url to use for an OpenAI compatible API")

    args = parser.parse_args()

    grade_score, llm_score, choice_score = stats_for_dataset(
        args.dataset,
        args.prompt,
        args.client,
        args.prompt_key,
        args.outputs_key,
        args.is_local,
        args.llm_bias_threshold,
        args.choice_consistency_threshold,
        args.negative_sampling,
        args.api_key,
        args.temperature,
        args.model,
        args.url
    )

    save_dict = {
        "grade_score": grade_score,
        "llm_score": llm_score,
        "choice_score": choice_score
    }

    with open(args.save_path, "w") as f:
        json.dump(save_dict, f)
