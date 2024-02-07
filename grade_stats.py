from typing import List

from scipy.stats import mode


def test_stability_selection(selected_indxes: List, total_options: int):
    # generate counts table for our selected options
    counts_actual = _build_counts(selected_indxes, total_options)

    # generate a mock distribution for how it'd look like given a uniform distribution
    uniform_per_item_amount = len(selected_indxes) // total_options
    ideal_count = [uniform_per_item_amount] * total_options
    return mannwhitneyu(selected_indxes, ideal_count)


def test_choice_consistency(actual_indexes: List, total_options: int):
    # generate observed counts table
    counts_actual = _build_counts(actual_indexes, total_options)
    most_common = mode(actual_indexes, axis=None, keepdims=False).mode
    return counts_actual[most_common] / len(actual_indexes)


def _build_counts(indexes: List, total_options: int):
    counts_actual = [0] * total_options
    for sel in indexes:
        counts_actual[sel] += 1

    return counts_actual

