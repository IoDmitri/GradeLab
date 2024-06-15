import argparse
import json
import os
from client import client_from_args
from llm_eval import Evaluator
from prompts import judge_prompt, judge_prompt_v2, judge_prompt_v3

from datasets import load_dataset, load_from_disk


def stats_for_dataset(dataset: str, prompt_path: str, client: str, prompt_key:str, outputs_key: str, is_local=False,
                      random_option=False, api_key=None, temperature=0.3, model=None, url=None):

    ds = load_from_disk(dataset) if is_local else load_dataset(dataset)
    if "train" in ds:
        ds = ds["train"]

    print(f"loaded dataset {dataset}")
    if client not in ["openai", "mistral", "anthropic", "together"]:
        raise ValueError(f"Expected 'openai' or 'mistral' or 'anthropic' or 'together', got {client}, which is currently unsupported")

    client = client_from_args(client, api_key=api_key, model=model, url=url)

    if not os.path.exists(prompt_path):
        raise ValueError(f"provided prompt path: {prompt_path} was not found")

    with open(prompt_path, "r") as f:
        judge_prompt_to_use = f.read()

    evaluator = Evaluator(client, judge_prompt_to_use)
    gen_args = {"temperature": temperature}
    if model:
        gen_args["model"] = model

    return evaluator.grade_stats_for_dataset(
        ds, prompt_key, outputs_key, judge_prompt_to_use, random_option=random_option, **gen_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate dataset with LLMs.")
    parser.add_argument("dataset", type=str, help="Name of the dataset or path if local.")
    parser.add_argument("prompt", type=str, help="Prompt type or custom prompt.")
    parser.add_argument("client", type=str, choices=["openai", "mistral", "anthropic"], help="Client to use: 'openai' or 'mistral' or 'anthropic'.")
    parser.add_argument("prompt_key", type=str, help="Key for the prompt in the dataset.")
    parser.add_argument("outputs_key", type=str, help="Key for the outputs in the dataset.")
    parser.add_argument("--is_local", action="store_true", help="Flag to load dataset from disk.")
    parser.add_argument("--random_option", action="store_true", help="Flag for using a random option.")
    parser.add_argument("--api_key", type=str, help="API key for the client.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for generation.")
    parser.add_argument("--model", type=str, help="Model to be used for generation.")
    parser.add_argument("--save_path", type=str, default="output.json", help="Path to save the output.")
    parser.add_argument("--url", type=str, help="the url to use for an OpenAI compatible API")

    args = parser.parse_args()

    mean_grade_score, mean_entropy, mean_choice_score, llm_sel_stats, llm_choice_stats = stats_for_dataset(
        args.dataset,
        args.prompt,
        args.client,
        args.prompt_key,
        args.outputs_key,
        args.is_local,
        args.random_option,
        args.api_key,
        args.temperature,
        args.model,
        args.url
    )

    save_dict = {
        "grade_score": mean_grade_score,
        "entropy": mean_entropy,
        "choice_score": mean_choice_score,
        "llm_selection_stats": llm_sel_stats,
        "llm_choice_stats": llm_choice_stats
    }

    with open(args.save_path, "w") as f:
        json.dump(save_dict, f)
