import json
import argparse

from datasets import load_dataset
from datasets import load_metric


def sharedtask1_metrics(prediction_json, split, cache_dir=None):
    metric = load_metric("squad_v2")

    predictions = json.load(open(prediction_json, "r"))
    d_id_prediction = {}
    for ele in predictions:
        d_id_prediction[ele["id"]] = 0

    references = []
    d_id_reference = {}
    dataset = load_dataset(
        "doc2dial",
        name="doc2dial_rc",
        split=split,
        ignore_verifications=True,
        cache_dir=cache_dir,
    )
    for ex in dataset:
        if ex["id"] not in d_id_prediction:
            continue
        d_id_reference[ex["id"]] = 0
        references.append(
            {
                "id": ex["id"],
                "answers": ex["answers"],
            }
        )
    assert (
        len(predictions)
        == len(references)
        == len(d_id_prediction)
        == len(d_id_reference)
    ), "Ensure the matching count of instances of references and predictioins"

    metric.add_batch(predictions=predictions, references=references)
    final_score = metric.compute()
    """
    print(final_score)
    OrderedDict([('exact', 33.333333333333336), ('f1', 38.095238095238095), ('span', 33.333333333333336), ('total', 3), ('HasAns_exact', 33.333333333333336), ('HasAns_f1', 38.095238095238095), ('HasAns_total', 3)])
    """
    return final_score


def sharedtask2_metrics(prediction_json, split, cache_dir):
    metric_sacrebleu = load_metric("sacrebleu")

    predictions = json.load(open(prediction_json, "r"))
    d_id_prediction = {}
    model_predictions = []
    for ex in predictions:
        model_predictions.append(ex["utterance"])
        d_id_prediction[ex["id"]] = 0

    references_lst = []
    dataset = load_dataset(
        "doc2dial",
        name="dialogue_domain",
        split=split,
        ignore_verifications=True,
        cache_dir=cache_dir,
    )
    for ex in dataset:
        for turn in ex["turns"]:
            if turn["role"] == "agent":
                id_ = "{}_{}".format(ex["dial_id"], turn["turn_id"] - 1)
                if id_ not in d_id_prediction:
                    continue
                references_lst.append([turn["utterance"]])
    assert (
        len(model_predictions) == len(references_lst) == len(d_id_prediction)
    ), "Ensure the matching count of instances of references and predictioins"
    metric_sacrebleu.add_batch(predictions=model_predictions, references=references_lst)
    final_score = metric_sacrebleu.compute()["score"]
    """
    print(final_score)
    sacrebleu 8.234381476893315
    """
    return final_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Select metrics for task that is either 'subtask1' or 'subtask2'",
    )
    parser.add_argument(
        "--prediction_json",
        type=str,
        required=True,
        help="Path to predictions",
    )
    parser.add_argument(
        "--split",
        default="validation",
        type=str,
        help='Data split for validation that is either "validation" or "test"',
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Path for caching the downloaded data by HuggingFace Datasets",
    )
    args = parser.parse_args()
    if args.task == "subtask1":
        sharedtask1_metrics(args.prediction_json, args.split, args.cache_dir)
    else:
        sharedtask2_metrics(args.prediction_json, args.split, args.cache_dir)


if __name__ == "__main__":
    main()
