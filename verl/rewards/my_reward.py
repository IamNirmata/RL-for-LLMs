# my_reward.py
import json

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    try:
        obj = json.loads(solution_str.strip())
    except Exception:
        return 0.0

    if "answer" not in obj:
        return 0.0

    if str(obj["answer"]).strip() == str(ground_truth).strip():
        return 1.0
    return 0.1
