from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("output/sapling_detection_eval_output_no_tokens.json")
DEFAULT_OUTPUT = Path("output/sapling_threshold_analysis.json")
DEFAULT_THRESHOLDS = (10.0, 20.0, 30.0, 40.0, 50.0)


def threshold_float(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold must be a number") from exc
    if value < 0 or value > 100:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 100")
    return value


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in {"human", "humans"}:
        return "human"
    if label in {"ai", "machine", "model"}:
        return "ai"
    return label


def normalize_percent(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        number = float(match.group(0))
        if "%" in text:
            return number
    else:
        return None

    if 0 <= number <= 1:
        return number * 100
    return number


def safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return round(2 * precision * recall / (precision + recall), 6)


def predicted_label(score_percent: float, threshold: float) -> str:
    if score_percent > threshold:
        return "ai"
    return "human"


def get_score_percent(item: dict[str, Any]) -> float | None:
    score_percent = normalize_percent(item.get("score_percent"))
    if score_percent is not None:
        return score_percent
    return normalize_percent(item.get("score"))


def load_records(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, dict):
        raise ValueError("Input JSON must contain an object field named items")

    skipped_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for item_key, item in items.items():
        if not isinstance(item, dict):
            skipped_counts["not_object"] += 1
            continue

        true_label = normalize_label(item.get("true_label"))
        if true_label not in {"human", "ai"}:
            skipped_counts["missing_true_label"] += 1
            continue

        score_percent = get_score_percent(item)
        if score_percent is None:
            skipped_counts["missing_score"] += 1
            continue

        records.append(
            {
                "key": item_key,
                "true_label": true_label,
                "score_percent": score_percent,
            }
        )

    return records, dict(skipped_counts)


def summarize_threshold(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    evaluated: list[tuple[str, str]] = []
    for record in records:
        evaluated.append((record["true_label"], predicted_label(record["score_percent"], threshold)))

    correct_count = sum(1 for true_label, predicted in evaluated if true_label == predicted)
    true_ai = sum(1 for true_label, _predicted in evaluated if true_label == "ai")
    true_human = sum(1 for true_label, _predicted in evaluated if true_label == "human")
    predicted_ai = sum(1 for _true_label, predicted in evaluated if predicted == "ai")
    predicted_human = sum(1 for _true_label, predicted in evaluated if predicted == "human")

    tp_ai = sum(1 for true_label, predicted in evaluated if true_label == "ai" and predicted == "ai")
    fp_ai = sum(1 for true_label, predicted in evaluated if true_label == "human" and predicted == "ai")
    fn_ai = sum(1 for true_label, predicted in evaluated if true_label == "ai" and predicted == "human")
    tn_ai = sum(1 for true_label, predicted in evaluated if true_label == "human" and predicted == "human")

    ai_precision = safe_divide(tp_ai, tp_ai + fp_ai)
    ai_recall = safe_divide(tp_ai, tp_ai + fn_ai)
    human_precision = safe_divide(tn_ai, tn_ai + fn_ai)
    human_recall = safe_divide(tn_ai, tn_ai + fp_ai)

    return {
        "threshold": threshold,
        "rule": f"score > {threshold}% => ai; score <= {threshold}% => human",
        "total": len(records),
        "correct": correct_count,
        "accuracy": safe_divide(correct_count, len(records)),
        "true_label_counts": {
            "ai": true_ai,
            "human": true_human,
        },
        "predicted_label_counts": {
            "ai": predicted_ai,
            "human": predicted_human,
        },
        "confusion_matrix_ai_positive": {
            "tp": tp_ai,
            "fp": fp_ai,
            "fn": fn_ai,
            "tn": tn_ai,
        },
        "ai_metrics": {
            "support": true_ai,
            "predicted": predicted_ai,
            "precision": ai_precision,
            "recall": ai_recall,
            "f1": f1_score(ai_precision, ai_recall),
        },
        "human_metrics": {
            "support": true_human,
            "predicted": predicted_human,
            "precision": human_precision,
            "recall": human_recall,
            "f1": f1_score(human_precision, human_recall),
        },
    }


def best_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accuracies = [result["accuracy"] for result in results if result.get("accuracy") is not None]
    if not accuracies:
        return []
    best_accuracy = max(accuracies)
    return [result for result in results if result.get("accuracy") == best_accuracy]


def ensure_parent_dir(path: Path) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def print_table(results: list[dict[str, Any]]) -> None:
    headers = [
        "threshold",
        "accuracy",
        "correct",
        "total",
        "pred_ai",
        "pred_human",
        "ai_precision",
        "ai_recall",
        "human_precision",
        "human_recall",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                format_number(result["threshold"]),
                format_number(result["accuracy"]),
                format_number(result["correct"]),
                format_number(result["total"]),
                format_number(result["predicted_label_counts"]["ai"]),
                format_number(result["predicted_label_counts"]["human"]),
                format_number(result["ai_metrics"]["precision"]),
                format_number(result["ai_metrics"]["recall"]),
                format_number(result["human_metrics"]["precision"]),
                format_number(result["human_metrics"]["recall"]),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find the best Sapling AI score threshold by accuracy.")
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT, help="Sapling evaluation JSON input")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="JSON file to save threshold analysis")
    parser.add_argument(
        "--threshold",
        action="append",
        type=threshold_float,
        default=[],
        help="Threshold percent to test. Repeat to test multiple values. Defaults to 10,20,30,40,50.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = args.threshold or list(DEFAULT_THRESHOLDS)
    thresholds = sorted(dict.fromkeys(thresholds))

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        records, skipped_counts = load_records(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not records:
        print("No evaluable records found", file=sys.stderr)
        return 1

    results = [summarize_threshold(records, threshold) for threshold in thresholds]
    winners = best_results(results)
    payload = {
        "input_file": str(args.input),
        "thresholds": thresholds,
        "evaluable_count": len(records),
        "skipped_counts": skipped_counts,
        "best_thresholds": [winner["threshold"] for winner in winners],
        "best_accuracy": winners[0]["accuracy"] if winners else None,
        "results": results,
    }

    ensure_parent_dir(args.output)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(f"Input: {args.input}")
    print(f"Evaluable records: {len(records)}")
    if skipped_counts:
        print(f"Skipped: {skipped_counts}")
    print_table(results)
    if winners:
        best_threshold_text = ", ".join(format_number(winner["threshold"]) for winner in winners)
        print(f"Best threshold(s): {best_threshold_text} with accuracy {format_number(winners[0]['accuracy'])}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())