from concurrent.futures import ThreadPoolExecutor
import random
import re
from typing import Iterable

from client import Client
from grade_score import grade_score
import prompts


def extract_option_number(text, sel_keyword="Selection"):
    # Start off by finding the `Selection` section

    sel_idx = text.find(sel_keyword)
    if sel_idx >= 0:
        text = text[sel_idx + len(sel_keyword):]
    # Search for 'Option' followed by a space and a number
    match = re.search(r'Option (\d+)', text)

    # If a match is found, return the number, otherwise return None
    if match:
        return int(match.group(1)) - 1


def unique_permutations(sq):
    prev_sq = sq
    for _ in range(len(sq)):
        prev_sq = [prev_sq[-1]] + prev_sq[0:-1]
        yield prev_sq


class Evaluator:
    def __init__(self, client: Client, judge_prompt=None):
        self.client = client
        self.judge_prompt = judge_prompt or prompts.judge_prompt

    def evaluate(self, outputs, instruction, judge_prompt=None, **generate_args):
        if not judge_prompt:
            judge_prompt = self.judge_prompt

        eval_input = f"""From the following outputs, make your selection. 
                  [User Instruction]
                  Instruction: {instruction}
                  [\\User Instruction]"""
        for idx, o in enumerate(outputs):
            eval_input += f"\n[Option {idx + 1}]\n\n {o} \n\n[\Option {idx + 1}]"
        if "temperature" not in generate_args:
            generate_args["temperature"] = 0.3
        eval_output = self.client.get_completion(judge_prompt, eval_input, **generate_args)
        return extract_option_number(eval_output)

    def _run_trial(self, outputs_with_idx, instruction, judge_prompt, **generate_args):
        # Extract shuffled list and the corresponding original indices
        shuffled_list = [item[1] for item in outputs_with_idx]
        original_indices = [item[0] for item in outputs_with_idx]
        # Run the evaluation function on the shuffled list
        selected_index = self.evaluate(shuffled_list, instruction, judge_prompt, **generate_args)
        true_winner_index = None
        if selected_index is not None and selected_index < len(outputs_with_idx):
            true_winner_index = original_indices[selected_index]
        return selected_index, true_winner_index

    def monte_carlo_evaluate(self, outputs, instruction, judge_prompt=None, **generate_args):
        shuffled_with_indices = list(enumerate(outputs))

        with ThreadPoolExecutor() as executor:
            # Map run_trial function to each trial
            results = list(executor.map(lambda output_set: self._run_trial(output_set,
                                                                           instruction,
                                                                           judge_prompt,
                                                                           **generate_args),
                                        unique_permutations(shuffled_with_indices)))

            return results

    def grade_stats_for_dataset(self, dataset: Iterable[dict], prompt_key, outputs_key, judge_prompt,
                                negative_sample=False, llm_bias_threshold=0.4, choice_consistency_threshold=0.6,
                                **generate_args):

        row_stats = []
        for row in dataset:
            outputs = row[outputs_key]
            prompt = row[prompt_key]

            if negative_sample:
                sample = random.choice(dataset)
                while sample[prompt_key] == prompt:
                    sample = random.choice(dataset)
                random_output = random.choice(sample[outputs_key])
                outputs = [random_output] + outputs

            monte_carl_results = self.monte_carlo_evaluate(outputs, prompt, judge_prompt=judge_prompt,
                                                           **generate_args)
            row_stats.append(monte_carl_results)

        return grade_score(row_stats)


