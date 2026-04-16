from verl.utils.reward_score.search_r1_like_qa_em import compute_score as search_r1_compute_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return float(search_r1_compute_score(solution_str, ground_truth))
