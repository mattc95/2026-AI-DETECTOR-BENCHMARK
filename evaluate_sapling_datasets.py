from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_ENDPOINT = "https://api.sapling.ai/api/v1/aidetect"
DEFAULT_DATASETS = (
    ("human_detection_test", Path("data/human_detection_test.json")),
    ("ai_detection_test", Path("data/ai_detection_test.json")),
)
DEFAULT_SCORE_THRESHOLD = 50.0
TOKEN_FIELDS = {"token_probs", "tokens"}


@dataclass(frozen=True)
class EvalJob:
    dataset: str
    item_id: str
    item: dict[str, Any]
    text: str
    true_label: str
    source: str
    category: str
    category_type: str
    word_count: int
    word_bucket: str


@dataclass(frozen=True)
class SaplingResult:
    status: str
    http_status: int | None
    score: Any
    score_path: str | None
    score_percent: float | None
    predicted_label: str | None
    response: Any
    error: str


def positive_int(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return value


def non_negative_int(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return value


def positive_float(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return value


def normalize_endpoint(endpoint: str) -> str:
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    return f"https://{endpoint}"


def parse_headers(raw_headers: list[str]) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    for raw_header in raw_headers:
        if "=" in raw_header:
            header_name, header_value = raw_header.split("=", 1)
        elif ":" in raw_header:
            header_name, header_value = raw_header.split(":", 1)
        else:
            raise ValueError(f"Invalid header {raw_header!r}; use Name=Value")

        header_name = header_name.strip()
        header_value = header_value.strip()
        if not header_name:
            raise ValueError(f"Invalid header {raw_header!r}; header name is empty")
        headers[header_name] = header_value
    return headers


def resolve_api_key(raw_key: str | None) -> str:
    key = raw_key if raw_key is not None else os.getenv("SAPLING_API_KEY", "")
    key = key.strip()
    if not key:
        raise ValueError("Sapling API key is required; pass --key or set SAPLING_API_KEY")
    return key


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in {"human", "humans"}:
        return "human"
    if label in {"ai", "machine", "model"}:
        return "ai"
    return label


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def word_bucket(count: int) -> str:
    if 50 <= count <= 200:
        return "50-200"
    if 200 < count <= 500:
        return "200-500"
    if 500 < count <= 1000:
        return "500-1000"
    return "other"


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


def score_to_label(score_percent: float | None, threshold: float = DEFAULT_SCORE_THRESHOLD) -> str | None:
    if score_percent is None:
        return None
    if score_percent > threshold:
        return "ai"
    return "human"


def score_bucket(score_percent: float | None) -> str:
    if score_percent is None:
        return "unknown"
    if score_percent <= 10:
        return "0-10"
    if score_percent <= 50:
        return "10-50"
    if score_percent <= 90:
        return "50-90"
    return "90-100"


def find_key(data: Any, wanted_key: str, path: tuple[str, ...] = ()) -> tuple[Any, str] | None:
    if isinstance(data, dict):
        if wanted_key in data:
            return data[wanted_key], ".".join((*path, wanted_key))
        for key, value in data.items():
            found = find_key(value, wanted_key, (*path, str(key)))
            if found is not None:
                return found
    elif isinstance(data, list):
        for index, value in enumerate(data):
            found = find_key(value, wanted_key, (*path, str(index)))
            if found is not None:
                return found
    return None


def extract_score(response_value: Any) -> tuple[Any, str | None, float | None]:
    if isinstance(response_value, dict) and "score" in response_value:
        raw_score = response_value.get("score")
        return raw_score, "score", normalize_percent(raw_score)

    found = find_key(response_value, "score")
    if found is None:
        return None, None, None
    raw_score, score_path = found
    return raw_score, score_path, normalize_percent(raw_score)


def remove_token_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: remove_token_fields(child) for key, child in value.items() if key not in TOKEN_FIELDS}
    if isinstance(value, list):
        return [remove_token_fields(item) for item in value]
    return value


def iter_json_items(data: Any):
    if isinstance(data, dict):
        for item_id, item in data.items():
            yield str(item_id), item
    elif isinstance(data, list):
        for index, item in enumerate(data, start=1):
            yield str(index), item
    else:
        raise ValueError("Input JSON must be an object or an array")


def get_text(raw_item: dict[str, Any], text_field: str) -> str:
    if text_field in {"", ".", "self"}:
        return str(raw_item or "").strip()
    return str(raw_item.get(text_field) or "").strip()


def load_dataset(dataset_name: str, input_path: Path, text_field: str) -> list[EvalJob]:
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    jobs: list[EvalJob] = []
    for item_id, raw_item in iter_json_items(data):
        if not isinstance(raw_item, dict):
            continue

        text = get_text(raw_item, text_field)
        true_label = normalize_label(raw_item.get("label"))
        if not text or true_label not in {"human", "ai"}:
            continue

        source = str(raw_item.get("source") or "").strip()
        if "theme" in raw_item:
            category = str(raw_item.get("theme") or "").strip()
            category_type = "theme"
        else:
            category = str(raw_item.get("style") or "").strip()
            category_type = "style"

        count = word_count(text)
        jobs.append(
            EvalJob(
                dataset=dataset_name,
                item_id=item_id,
                item=raw_item,
                text=text,
                true_label=true_label,
                source=source,
                category=category,
                category_type=category_type,
                word_count=count,
                word_bucket=word_bucket(count),
            )
        )
    return jobs


def parse_dataset_specs(raw_specs: list[str]) -> list[tuple[str, Path]]:
    if not raw_specs:
        return list(DEFAULT_DATASETS)

    datasets: list[tuple[str, Path]] = []
    for raw_spec in raw_specs:
        if "=" in raw_spec:
            name, raw_path = raw_spec.split("=", 1)
            dataset_name = name.strip()
            dataset_path = Path(raw_path.strip())
        else:
            dataset_path = Path(raw_spec.strip())
            dataset_name = dataset_path.stem
        if not dataset_name:
            raise ValueError(f"Invalid dataset spec {raw_spec!r}; dataset name is empty")
        datasets.append((dataset_name, dataset_path))
    return datasets


def build_request_body(key: str, text: str, score_string: bool) -> dict[str, Any]:
    return {
        "key": key,
        "text": text,
        "score_string": score_string,
    }


def classify_text(
    job: EvalJob,
    endpoint: str,
    key: str,
    headers: dict[str, str],
    timeout: float,
    retries: int,
    score_string: bool,
    score_threshold: float,
    drop_token_fields: bool,
) -> SaplingResult:
    last_error = ""
    last_http_status: int | None = None
    last_response: Any = None

    for attempt_number in range(retries + 1):
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=build_request_body(key, job.text, score_string),
                timeout=timeout,
            )
            last_http_status = response.status_code
            try:
                response_value = response.json()
            except ValueError:
                response_value = response.text
            if drop_token_fields:
                response_value = remove_token_fields(response_value)
            last_response = response_value
            response.raise_for_status()

            score, score_path, score_percent = extract_score(response_value)
            predicted_label = score_to_label(score_percent, score_threshold)
            if not isinstance(response_value, dict):
                return SaplingResult(
                    status="invalid_json",
                    http_status=response.status_code,
                    score=score,
                    score_path=score_path,
                    score_percent=score_percent,
                    predicted_label=predicted_label,
                    response=response_value,
                    error="Response JSON is not an object",
                )
            if score_percent is None:
                return SaplingResult(
                    status="missing_score",
                    http_status=response.status_code,
                    score=score,
                    score_path=score_path,
                    score_percent=None,
                    predicted_label=None,
                    response=response_value,
                    error="Could not find numeric score in response",
                )

            return SaplingResult(
                status="ok",
                http_status=response.status_code,
                score=score,
                score_path=score_path,
                score_percent=score_percent,
                predicted_label=predicted_label,
                response=response_value,
                error="",
            )
        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt_number < retries:
                time.sleep(min(2**attempt_number, 10))

    score, score_path, score_percent = extract_score(last_response)
    return SaplingResult(
        status="error",
        http_status=last_http_status,
        score=score,
        score_path=score_path,
        score_percent=score_percent,
        predicted_label=score_to_label(score_percent, score_threshold),
        response=last_response,
        error=last_error,
    )


def output_key(job: EvalJob) -> str:
    return f"{job.dataset}:{job.item_id}"


def response_field(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    return None


def build_output_item(job: EvalJob, result: SaplingResult) -> dict[str, Any]:
    predicted_label = result.predicted_label
    correct = predicted_label == job.true_label if predicted_label is not None else None
    return {
        "dataset": job.dataset,
        "id": job.item_id,
        "text": job.text,
        "source": job.source,
        "prompt": job.item.get("prompt", ""),
        "theme": job.item.get("theme", ""),
        "style": job.item.get("style", ""),
        "category": job.category,
        "category_type": job.category_type,
        "true_label": job.true_label,
        "word_count": job.word_count,
        "word_bucket": job.word_bucket,
        "status": result.status,
        "http_status": result.http_status,
        "score": result.score,
        "score_path": result.score_path,
        "score_percent": result.score_percent,
        "score_bucket": score_bucket(result.score_percent),
        "predicted_label": predicted_label,
        "correct": correct,
        "sentence_scores": response_field(result.response, "sentence_scores"),
        "sapling_response": result.response,
        "response": result.response,
        "error": result.error,
    }


def safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return round(2 * precision * recall / (precision + recall), 6)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(record.get("status")) for record in records)
    true_counts = Counter(str(record.get("true_label")) for record in records if record.get("true_label"))
    predicted_counts = Counter(str(record.get("predicted_label")) for record in records if record.get("predicted_label"))
    score_bucket_counts = Counter(str(record.get("score_bucket")) for record in records if record.get("score_bucket"))

    evaluable = [record for record in records if record.get("predicted_label") in {"human", "ai"}]
    correct_count = sum(1 for record in evaluable if record.get("correct") is True)

    true_ai = sum(1 for record in evaluable if record.get("true_label") == "ai")
    true_human = sum(1 for record in evaluable if record.get("true_label") == "human")
    predicted_ai = sum(1 for record in evaluable if record.get("predicted_label") == "ai")
    predicted_human = sum(1 for record in evaluable if record.get("predicted_label") == "human")

    tp_ai = sum(1 for record in evaluable if record.get("true_label") == "ai" and record.get("predicted_label") == "ai")
    fp_ai = sum(1 for record in evaluable if record.get("true_label") == "human" and record.get("predicted_label") == "ai")
    fn_ai = sum(1 for record in evaluable if record.get("true_label") == "ai" and record.get("predicted_label") == "human")
    tn_ai = sum(1 for record in evaluable if record.get("true_label") == "human" and record.get("predicted_label") == "human")

    ai_precision = safe_divide(tp_ai, tp_ai + fp_ai)
    ai_recall = safe_divide(tp_ai, tp_ai + fn_ai)
    human_precision = safe_divide(tn_ai, tn_ai + fn_ai)
    human_recall = safe_divide(tn_ai, tn_ai + fp_ai)

    return {
        "total": len(records),
        "evaluable": len(evaluable),
        "correct": correct_count,
        "accuracy": safe_divide(correct_count, len(evaluable)),
        "status_counts": dict(status_counts),
        "true_label_counts": dict(true_counts),
        "predicted_label_counts": dict(predicted_counts),
        "score_bucket_counts": dict(score_bucket_counts),
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


def grouped_metrics(records: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_value = str(record.get(field_name) or "unknown")
        groups[group_value].append(record)
    return {group_value: summarize_records(group_records) for group_value, group_records in sorted(groups.items())}


def build_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": summarize_records(records),
        "by_dataset": grouped_metrics(records, "dataset"),
        "by_true_label": grouped_metrics(records, "true_label"),
        "by_word_bucket": grouped_metrics(records, "word_bucket"),
        "by_source": grouped_metrics(records, "source"),
        "by_category": grouped_metrics(records, "category"),
        "by_category_type": grouped_metrics(records, "category_type"),
        "by_score_bucket": grouped_metrics(records, "score_bucket"),
    }


def load_existing_items(output_path: Path) -> dict[str, Any]:
    if not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    items = data.get("items") if isinstance(data, dict) else None
    if isinstance(items, dict):
        return items
    return {}


def ensure_parent_dir(path: Path) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def iter_batches(items: list[EvalJob], batch_size: int):
    for start_index in range(0, len(items), batch_size):
        yield items[start_index : start_index + batch_size]


def save_output(
    output_path: Path,
    endpoint: str,
    dataset_specs: list[tuple[str, Path]],
    text_field: str,
    score_threshold: float,
    score_string: bool,
    drop_token_fields: bool,
    items: dict[str, Any],
) -> None:
    records = list(items.values())
    payload = {
        "meta": {
            "endpoint": endpoint,
            "datasets": [{"name": name, "path": str(path)} for name, path in dataset_specs],
            "input_text_field": text_field,
            "request_field": "text",
            "request_options": {
                "score_string": score_string,
                "drop_token_fields": drop_token_fields,
            },
            "prediction_rule": f"Sapling score > {score_threshold}% => ai; score <= {score_threshold}% => human",
            "include_response": True,
            "item_count": len(records),
        },
        "metrics": build_metrics(records),
        "items": items,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sapling AI detector on human/AI test JSON files and evaluate metrics.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset spec as name=path. Defaults to data/human_detection_test.json and data/ai_detection_test.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/sapling_detection_eval_output.json",
        help="Single JSON output file for Sapling results and metrics.",
    )
    parser.add_argument("--field", default="text", help="Input JSON field sent to Sapling as text")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Sapling AI detector endpoint")
    parser.add_argument("--key", default=None, help="Sapling API key. Defaults to SAPLING_API_KEY.")
    parser.add_argument("--timeout", type=positive_float, default=60.0, help="Request timeout in seconds")
    parser.add_argument("--retries", type=non_negative_int, default=3, help="Retry count per item")
    parser.add_argument(
        "-n",
        "--rate-per-second",
        type=positive_int,
        default=5,
        help="Maximum request count started per second",
    )
    parser.add_argument("--save-every", type=positive_int, default=10, help="Save progress every N processed items")
    parser.add_argument("--threshold", type=positive_float, default=DEFAULT_SCORE_THRESHOLD, help="AI threshold in percent")
    parser.add_argument("--score-string", action="store_true", help="Request Sapling score_string=true")
    parser.add_argument("--drop-token-fields", action="store_true", help="Do not save token_probs or tokens fields in responses")
    parser.add_argument("--header", action="append", default=[], help="Extra HTTP header, such as 'X-Name=value'")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from output file")
    resume_group.add_argument("--restart", dest="resume", action="store_false", help="Ignore existing output and start over")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dataset_specs = parse_dataset_specs(args.dataset)
        headers = parse_headers(args.header)
        key = resolve_api_key(args.key)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_path = Path(args.output)
    endpoint = normalize_endpoint(args.endpoint)
    ensure_parent_dir(output_path)

    all_jobs: list[EvalJob] = []
    for dataset_name, dataset_path in dataset_specs:
        if not dataset_path.exists():
            print(f"Input file not found: {dataset_path}", file=sys.stderr)
            return 1
        dataset_jobs = load_dataset(dataset_name, dataset_path, args.field)
        all_jobs.extend(dataset_jobs)
        print(f"Loaded {dataset_path}: {len(dataset_jobs)} valid items")

    output_items = load_existing_items(output_path) if args.resume else {}
    jobs = [job for job in all_jobs if not (args.resume and output_items.get(output_key(job), {}).get("status") == "ok")]
    resumed_count = len(all_jobs) - len(jobs)

    print(f"Endpoint: {endpoint}")
    print(f"Output: {output_path}")
    print(f"Input JSON field: {args.field}")
    print("Request field: text")
    print(f"Items to request: {len(jobs)}, resumed: {resumed_count}")
    print(f"Rule: Sapling score > {args.threshold}% => ai; score <= {args.threshold}% => human")

    processed_count = 0
    saved_count = 0
    started_at = time.monotonic()

    try:
        for batch_number, batch in enumerate(iter_batches(jobs, args.rate_per_second), start=1):
            batch_started_at = time.monotonic()
            batch_results: dict[str, SaplingResult] = {}

            with ThreadPoolExecutor(max_workers=args.rate_per_second) as executor:
                future_to_job = {
                    executor.submit(
                        classify_text,
                        job,
                        endpoint,
                        key,
                        headers,
                        args.timeout,
                        args.retries,
                        args.score_string,
                        args.threshold,
                        args.drop_token_fields,
                    ): job
                    for job in batch
                }
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    batch_results[output_key(job)] = future.result()

            for job in batch:
                key_name = output_key(job)
                output_items[key_name] = build_output_item(job, batch_results[key_name])
                processed_count += 1

            if args.save_every and processed_count - saved_count >= args.save_every:
                save_output(
                    output_path,
                    endpoint,
                    dataset_specs,
                    args.field,
                    args.threshold,
                    args.score_string,
                    args.drop_token_fields,
                    output_items,
                )
                saved_count = processed_count

            elapsed_seconds = time.monotonic() - started_at
            current_metrics = build_metrics(list(output_items.values()))["overall"]
            print(
                f"Batch {batch_number}: processed {processed_count}/{len(jobs)} new, "
                f"ok={current_metrics['status_counts'].get('ok', 0)}, "
                f"accuracy={current_metrics['accuracy']} in {elapsed_seconds:.1f}s"
            )

            batch_elapsed_seconds = time.monotonic() - batch_started_at
            if processed_count < len(jobs) and batch_elapsed_seconds < 1.0:
                time.sleep(1.0 - batch_elapsed_seconds)
    except KeyboardInterrupt:
        save_output(
            output_path,
            endpoint,
            dataset_specs,
            args.field,
            args.threshold,
            args.score_string,
            args.drop_token_fields,
            output_items,
        )
        print(f"Interrupted: progress saved to {output_path}", file=sys.stderr)
        return 130

    save_output(
        output_path,
        endpoint,
        dataset_specs,
        args.field,
        args.threshold,
        args.score_string,
        args.drop_token_fields,
        output_items,
    )
    final_metrics = build_metrics(list(output_items.values()))["overall"]
    print("Done")
    print(f"Output JSON: {output_path}")
    print(f"Total: {final_metrics['total']}")
    print(f"Evaluable: {final_metrics['evaluable']}")
    print(f"Accuracy: {final_metrics['accuracy']}")
    print(f"AI metrics: {final_metrics['ai_metrics']}")
    print(f"Human metrics: {final_metrics['human_metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())