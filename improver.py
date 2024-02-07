import argparse
import json

from client import client_from_args
from prompts import judge_prompt, judge_prompt_v2, judge_prompt_v3
from experiment import stats_for_dataset


def extract_judge_prompt(improver_output):
    index = improver_output.find('Output:') + len('Output: ')
    extracted_text = improver_output[index:].strip()
    return extracted_text


class Improver:
    def __init__(self, improver_prompt: str, improver_client: str, **client_args):
        self.system_prompt = improver_prompt
        self.client = client_from_args(improver_client, **client_args)

    def improve_prompt(self, initial_judge_prompt: str, dataset_str: str, prompt_key: str, outputs_key: str,
                       api_key: str, grader_client_type: str, grade_model: str, is_local=True, trials=3):
        grading_outputs = []

        current_judge_prompt = initial_judge_prompt

        for trial in range(trials):
            grade_score, llm_score, choice_score = stats_for_dataset(
                dataset_str,
                current_judge_prompt,
                grader_client_type,
                prompt_key,
                outputs_key,
                is_local,
                api_key,
                model=grade_model
            )

            print(f"tr: {trial} | grade score: {grade_score} | llm_score : {llm_score} | choice_score : {choice_score}")
            improver_message = f"{{judge_prompt: {current_judge_prompt}, grade_score: {grade_score}}}"
            improver_output = self.client.get_completion(system=self.system_prompt, message=improver_message)
            print(f"full improver output:\n {improver_output}")
            next_judge_prompt = extract_judge_prompt(improver_output)
            if next_judge_prompt:
                print(f"next judge prompt: {next_judge_prompt}")
                current_judge_prompt = next_judge_prompt
            grading_outputs.append((grade_score, next_judge_prompt, improver_output))

        grading_outputs.sort(key=lambda x: x[0], reverse=True)
        prompt_scores = [item[0] for item in grading_outputs]
        judge_prompts = [item[1] for item in grading_outputs]
        improver_messages = [item[2] for item in grading_outputs]

        return judge_prompts, prompt_scores, improver_messages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Improve the judgement prompt by using the grade score and a dataset.")
    parser.add_argument("dataset", type=str, help="Name of the dataset or path if local.")
    parser.add_argument("improver_prompt", type=str, help="Prompt type or custom prompt. Expects a file.")
    parser.add_argument("improver_client", type=str, choices=["openai", "mistral"],
                        help="Client to use for improver: 'openai' or 'mistral'.")
    parser.add_argument("judge_prompt", type=str, help="the initial grading prompt to be otpimized")
    parser.add_argument("grade_client", type=str, choices=["openai", "mistral"],
                        help="Client to use for grader: 'openai' or 'mistral'.")
    parser.add_argument("prompt_key", type=str, help="Key for the prompt in the dataset.")
    parser.add_argument("outputs_key", type=str, help="Key for the outputs in the dataset.")
    parser.add_argument("--is_local", action="store_true", help="Flag to load dataset from disk.")
    parser.add_argument("--llm_bias_threshold", type=float, default=0.4, help="LLM bias threshold.")
    parser.add_argument("--choice_consistency_threshold", type=float, default=0.6,
                        help="Choice consistency threshold.")
    parser.add_argument("--negative_sampling", action="store_true", help="Flag for negative sampling.")
    parser.add_argument("--api_key", type=str, help="API key for the client.")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for generation.")
    parser.add_argument("--improver_model", type=str, help="Model to be used for generation for the improver.")
    parser.add_argument("--grade_model", type=str, help="Model to be used for grading outputs.")
    parser.add_argument("--save_path", type=str, default="output.json", help="Path to save the output.")
    parser.add_argument("--trials", type=int, default=3, help="how many iterations to run for prompt improvement")
    parser.add_argument("--output_path", type=str, default="best_prompt.txt", )

    args = parser.parse_args()

    improver_prompt_text = None
    with open(args.improver_prompt, "r") as f:
        improver_prompt_text = f.read()

    improver = Improver(improver_prompt_text, args.improver_client, model=args.improver_model, api_key=args.api_key)
    judge_prompts, prompt_scores, improver_messages = improver.improve_prompt(
        args.judge_prompt,
        args.dataset,
        args.prompt_key,
        args.outputs_key,
        args.api_key,
        args.grade_client,
        args.grade_model,
        args.args.is_local,
        args.trials)

    with open(args.output_path, "w") as f:
        f.write(judge_prompts[0])















