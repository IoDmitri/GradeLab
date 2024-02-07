from scipy.stats import mode
import numpy as np


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