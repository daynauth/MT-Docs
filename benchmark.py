from typing import List, Union
from Levenshtein import distance as edit_distance
from tqdm import tqdm
import json

from utils import read_json

def get_single_kie_metric(prediction: dict[str, str], ground_truth: dict[str, str]) -> float:
    """
    Compute Levenshtein similarity per prediction (sample-level),
    averaged across fields, and return:
        - overall average score
        - list of per-sample scores
    """

    field_scores: list[float] = []

    for key, true_value in ground_truth.items():
        pred_value = prediction.get(key, "")

        gt_value = ground_truth[key]
        assert gt_value is not None, f"Ground truth value for key '{key}' is None"

        distance = edit_distance(pred_value, gt_value)
        max_len = max(len(pred_value), len(gt_value))

        if max_len == 0:
            field_scores.append(1.0)
        else:
            field_scores.append(1 - distance / max_len)

    return sum(field_scores) / len(field_scores) if field_scores else 0.0


def get_kie_metrics(start: int = 0, end: int = 100) -> float:
    sample_scores = []

    for index in tqdm(range(start, end)):
        platform = "litellm"
        gt_file = f"./data/annotations/{index}.json"
        pred_file = f"./results/{platform}/gemma3_12b_it_qat/{index}.json"

        gt = read_json(gt_file)
        pred = read_json(pred_file)
        sample_avg = get_single_kie_metric(pred, gt)
        sample_scores.append(sample_avg)

    overall_avg = sum(sample_scores) / len(sample_scores)
    return overall_avg

if __name__ == "__main__":
    overall_kie_score = get_kie_metrics(0, 10)
    print(f"Overall KIE Score: {overall_kie_score:.4f}")

