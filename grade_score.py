from scipy.stats import mode
import numpy as np


def compute_most_common_percentage(array):
    # Find the most common element
    most_common = np.bincount(array).argmax()

    # Count the frequency of the most common element
    count_most_common = np.sum(array == most_common)

    # Compute its percentage
    percentage_most_common = count_most_common / len(array)

    return percentage_most_common


def compute_entropy(selections, num_possible_choices):
    # Count the frequency of each selection
    freq, _ = np.histogram(selections, bins=np.arange(num_possible_choices + 1))

    # Calculate the probability of each selection
    total_selections = len(selections)
    probabilities = freq / total_selections

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities, where=(probabilities != 0)))

    # Normalize entropy to range [0, 1]
    max_entropy = np.log2(num_possible_choices)
    normalized_entropy = entropy / max_entropy

    return normalized_entropy


def llm_judge_grade_score(row):
    total_options = len(row)
    usable_rows = [r for r in row if None not in r]
    if usable_rows:
        llm_sel = [r[0] for r in usable_rows]
        choice_sel = [r[1] for r in usable_rows]

        llm_entropy = compute_entropy(llm_sel, total_options)
        choice_score = compute_most_common_percentage(choice_sel)

        return (2 * (llm_entropy * choice_score)) / (llm_entropy + choice_score), \
            llm_entropy, \
            choice_score, \
            llm_sel, \
            choice_sel
    return 0.0, 0.0, 0.0, [], []


def llm_grade_stats(row):
    row_stats = [llm_judge_grade_score(r) for r in row]
    mean_grade_score = np.mean([r_stats[0] for r_stats in row_stats])
    mean_entropy = np.mean([r_stats[1] for r_stats in row_stats])
    mean_choice_score = np.mean([r_stats[2] for r_stats in row_stats])

    llm_sel_stats, llm_choice_stats = array_prob_stats([r_stats[3] for r_stats in row_stats],
                                                       [r_stats[4] for r_stats in row_stats])

    return mean_grade_score, mean_entropy, mean_choice_score, llm_sel_stats, llm_choice_stats


def array_prob_stats(selected_indices, original_positions):
    if not selected_indices:
        return []

    # Flatten the lists using a list comprehension
    flat_llm_select = np.array([item for sublist in selected_indices for item in sublist])
    flat_choice_sel = np.array([item for sublist in original_positions for item in sublist])

    # Determine the maximum index to account for all possible selections
    max_index = max(flat_llm_select.max(), flat_choice_sel.max()) + 1

    # Calculate probabilities for llm_select and choice_sel
    prob_selected = _calculate_probabilities(flat_llm_select, max_index)
    prob_original = _calculate_probabilities(flat_choice_sel, max_index)

    return prob_selected, prob_original


def _calculate_probabilities(flat_array, max_index):
    counts = np.zeros(max_index)
    for value in flat_array:
        counts[value] += 1
    probabilities = counts / len(flat_array)
    return probabilities.tolist()

