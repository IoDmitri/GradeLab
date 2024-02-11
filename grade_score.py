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

        return llm_entropy * choice_score
    return 0.0


def grade_score(row_stats):
    return np.mean([llm_judge_grade_score(row) for row in row_stats])


def llm_judge_stats_comp(row):
    usable_rows = [r for r in row if None not in r]
    mode_stats = mode(usable_rows)
    if not isinstance(mode_stats.count, np.ndarray):
        return None
    llm_mode = mode_stats.mode[0]
    selected_mode = mode_stats.mode[1]
    llm_idx_sel_pct = mode_stats.count[0] / len(usable_rows)
    selected_idx_sel_pct = mode_stats.count[1] / len(usable_rows)
    return llm_mode, selected_mode, llm_idx_sel_pct, selected_idx_sel_pct


def score_grading(llm_stats, llm_bias_threshold=0.4, choice_consistency_thereshold=0.6):
    llm_score = 0
    choice_score = 0

    for stat in llm_stats:
        if not stat:
            continue
        _, _, llm_sel_pctg, choice_sel_pctg = stat

        if llm_sel_pctg <= llm_bias_threshold:
            llm_score += 1

        if choice_sel_pctg >= choice_consistency_thereshold:
            choice_score += 1

    llm_score = llm_score / len(llm_stats)
    choice_score = choice_score / len(llm_stats)
    grade_score = (2 * llm_score * choice_score) / (llm_score + choice_score)
    return grade_score, llm_score, choice_score