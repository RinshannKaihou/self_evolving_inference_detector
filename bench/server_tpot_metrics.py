#!/usr/bin/env python3
"""Read vLLM request-timing histograms from its Prometheus endpoint.

The timing values emitted by vLLM 0.27.0 are cumulative Prometheus histogram
``_sum`` and ``_count`` samples.  This module deliberately reports differences
between two raw ``/metrics`` snapshots; the benchmark runner owns snapshot
storage and the policy for checking its expected request count.

Only the vLLM 0.27.0 labels used by these histograms are accepted:
``model_name`` and ``engine``.  A zero-valued sample is distinct from an absent
sample.  In particular, a snapshot taken before the metric labels exist is not
silently treated as a zero baseline.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Final
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


TIMING_METRICS: Final[tuple[str, ...]] = (
    "vllm:request_time_per_output_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:time_to_first_token_seconds",
)
"""The vLLM 0.27.0 histogram base names included in a summary."""

_GROUP_LABELS: Final[frozenset[str]] = frozenset({"engine", "model_name"})
_SAMPLE_NAME_RE: Final = re.compile(r"^[A-Za-z_:][A-Za-z0-9_:]*")
_SAMPLE_RE: Final = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?"
    r"\s+(?P<value>\S+)"
    r"(?:\s+\S+)?\s*$"
)
_LABEL_NAME_RE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class MetricsParseError(ValueError):
    """Raised when a selected Prometheus sample cannot be parsed safely."""


Labels = tuple[tuple[str, str], ...]
GroupKey = tuple[str, str]


def fetch_metrics(base_url: str, timeout: float = 5.0) -> str:
    """Fetch raw Prometheus text from ``base_url + '/metrics'``.

    Passing an already-complete ``.../metrics`` URL is also supported.  HTTP
    errors and connection errors are deliberately propagated to the runner.
    """
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout!r}")

    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"base_url must be an absolute HTTP(S) URL, got {base_url!r}")
    path = parsed.path.rstrip("/")
    if not path.endswith("/metrics"):
        path = f"{path}/metrics" if path else "/metrics"
    endpoint = urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))
    request = Request(endpoint, headers={"Accept": "text/plain; version=0.0.4"})
    with urlopen(request, timeout=timeout) as response:  # nosec B310: caller owns URL
        return response.read().decode("utf-8")


def metric_total(text: str, full_sample_name: str) -> float | None:
    """Return the selected Prometheus sample total across all labels.

    ``None`` means that the sample is absent.  ``0.0`` means it is present and
    its label-series sum is zero, which lets a runner distinguish a new/empty
    histogram from an unavailable metrics family.  Duplicate label series and
    malformed selected samples raise :class:`MetricsParseError`.
    """
    samples = _parse_selected_samples(text, {full_sample_name})
    values = samples[full_sample_name]
    return math.fsum(values.values()) if values else None


def summarize_metrics_delta(before_text: str, after_text: str) -> dict:
    """Summarize cumulative vLLM timing histograms between two snapshots.

    The result has per-``model_name``/``engine`` rows in ``by_engine_model``
    and a label-aggregated ``total``.  Every ``sum`` and ``count`` in these
    result rows is a *delta*; ``mean`` is ``sum / count`` or ``None`` for a
    zero delta count.  The corresponding cumulative source values are retained
    in ``before`` and ``after`` for auditability.

    This function does not invent a zero baseline for an absent selected
    sample.  It returns ``valid=False`` with explicit errors for missing
    samples, label mismatches, decreasing values (service/metric reset),
    negative values, non-integral counts, or duplicate label series.  A caller
    should reject an invalid result before reporting it.
    """
    errors: list[str] = []
    warnings: list[str] = []
    try:
        before = _timing_snapshot(before_text)
    except MetricsParseError as exc:
        before = {}
        errors.append(f"before snapshot parse error: {exc}")
    try:
        after = _timing_snapshot(after_text)
    except MetricsParseError as exc:
        after = {}
        errors.append(f"after snapshot parse error: {exc}")

    groups = sorted(set(before) | set(after))
    if not groups:
        errors.append(
            "no selected vLLM timing histogram samples found in either snapshot"
        )

    rows: list[dict] = []
    for group in groups:
        row_metrics: dict[str, dict] = {}
        group_valid = True
        for metric in TIMING_METRICS:
            before_values = before.get(group, {}).get(metric, {})
            after_values = after.get(group, {}).get(metric, {})
            summary, metric_errors = _summarize_one_metric(
                metric, group, before_values, after_values
            )
            row_metrics[metric] = summary
            if metric_errors:
                group_valid = False
                errors.extend(metric_errors)
        rows.append(
            {
                "labels": {"model_name": group[0], "engine": group[1]},
                "valid": group_valid,
                "metrics": row_metrics,
            }
        )

    total = _summarize_total(rows)
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "by_engine_model": rows,
        "total": total,
    }


def _summarize_one_metric(
    metric: str,
    group: GroupKey,
    before_values: dict[str, float],
    after_values: dict[str, float],
) -> tuple[dict, list[str]]:
    errors: list[str] = []
    label_description = _format_group(group)
    before_present = {part: part in before_values for part in ("sum", "count")}
    after_present = {part: part in after_values for part in ("sum", "count")}

    for snapshot, present in (("before", before_present), ("after", after_present)):
        for part, is_present in present.items():
            if not is_present:
                errors.append(
                    f"missing {metric}_{part} in {snapshot} snapshot for {label_description}"
                )

    before_sum = before_values.get("sum")
    before_count = before_values.get("count")
    after_sum = after_values.get("sum")
    after_count = after_values.get("count")
    if not errors:
        assert before_sum is not None and before_count is not None
        assert after_sum is not None and after_count is not None
        for snapshot, value, description in (
            ("before", before_sum, "sum"),
            ("before", before_count, "count"),
            ("after", after_sum, "sum"),
            ("after", after_count, "count"),
        ):
            if not math.isfinite(value) or value < 0:
                errors.append(
                    f"invalid {metric}_{description}={value!r} in {snapshot} snapshot "
                    f"for {label_description}; expected a finite non-negative value"
                )
        for snapshot, count in (("before", before_count), ("after", after_count)):
            if math.isfinite(count) and count >= 0 and not count.is_integer():
                errors.append(
                    f"invalid {metric}_count={count!r} in {snapshot} snapshot for "
                    f"{label_description}; histogram count must be integral"
                )
        if not errors and (after_sum < before_sum or after_count < before_count):
            errors.append(
                f"counter reset detected for {metric} at {label_description}: "
                f"before(sum={before_sum}, count={before_count}), "
                f"after(sum={after_sum}, count={after_count})"
            )

    valid = not errors
    delta_sum = after_sum - before_sum if valid else None
    delta_count = after_count - before_count if valid else None
    return (
        {
            "valid": valid,
            "sum": delta_sum,
            "count": delta_count,
            "mean": (delta_sum / delta_count if delta_count else None),
            "before": {
                "present": all(before_present.values()),
                "sum": before_sum,
                "count": before_count,
            },
            "after": {
                "present": all(after_present.values()),
                "sum": after_sum,
                "count": after_count,
            },
        },
        errors,
    )


def _summarize_total(rows: Iterable[dict]) -> dict[str, dict]:
    rows = list(rows)
    result: dict[str, dict] = {}
    for metric in TIMING_METRICS:
        metric_rows = [row["metrics"][metric] for row in rows]
        valid = bool(metric_rows) and all(item["valid"] for item in metric_rows)
        if valid:
            total_sum = math.fsum(item["sum"] for item in metric_rows)
            total_count = math.fsum(item["count"] for item in metric_rows)
            result[metric] = {
                "valid": True,
                "sum": total_sum,
                "count": total_count,
                "mean": total_sum / total_count if total_count else None,
            }
        else:
            result[metric] = {
                "valid": False,
                "sum": None,
                "count": None,
                "mean": None,
            }
    return result


def _timing_snapshot(text: str) -> dict[GroupKey, dict[str, dict[str, float]]]:
    wanted = {
        f"{metric}_{part}" for metric in TIMING_METRICS for part in ("sum", "count")
    }
    samples = _parse_selected_samples(text, wanted)
    snapshot: dict[GroupKey, dict[str, dict[str, float]]] = {}
    for metric in TIMING_METRICS:
        for part in ("sum", "count"):
            sample_name = f"{metric}_{part}"
            for labels, value in samples[sample_name].items():
                group = _group_from_labels(labels, sample_name)
                snapshot.setdefault(group, {}).setdefault(metric, {})[part] = value
    return snapshot


def _parse_selected_samples(
    text: str, selected_names: set[str]
) -> dict[str, dict[Labels, float]]:
    parsed = {name: {} for name in selected_names}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name_match = _SAMPLE_NAME_RE.match(line)
        if name_match is None or name_match.group(0) not in selected_names:
            continue
        match = _SAMPLE_RE.fullmatch(line)
        if match is None:
            raise MetricsParseError(
                f"line {line_number}: malformed selected sample {raw_line!r}"
            )
        name = match.group("name")
        labels = _parse_labels(match.group("labels") or "", line_number)
        try:
            value = float(match.group("value"))
        except ValueError as exc:
            raise MetricsParseError(
                f"line {line_number}: invalid value for {name}: {match.group('value')!r}"
            ) from exc
        key = tuple(sorted(labels.items()))
        if key in parsed[name]:
            raise MetricsParseError(
                f"line {line_number}: duplicate {name} label series {dict(key)!r}"
            )
        parsed[name][key] = value
    return parsed


def _parse_labels(source: str, line_number: int) -> dict[str, str]:
    if not source:
        return {}
    labels: dict[str, str] = {}
    index = 0
    while index < len(source):
        while index < len(source) and source[index].isspace():
            index += 1
        match = _LABEL_NAME_RE.match(source, index)
        if match is None:
            raise MetricsParseError(f"line {line_number}: malformed label name")
        name = match.group(0)
        index = match.end()
        while index < len(source) and source[index].isspace():
            index += 1
        if index >= len(source) or source[index] != "=":
            raise MetricsParseError(f"line {line_number}: label {name!r} has no equals sign")
        index += 1
        while index < len(source) and source[index].isspace():
            index += 1
        if index >= len(source) or source[index] != '"':
            raise MetricsParseError(f"line {line_number}: label {name!r} is not quoted")
        index += 1
        value, index = _parse_quoted_label_value(source, index, line_number)
        if name in labels:
            raise MetricsParseError(f"line {line_number}: duplicate label {name!r}")
        labels[name] = value
        while index < len(source) and source[index].isspace():
            index += 1
        if index == len(source):
            break
        if source[index] != ",":
            raise MetricsParseError(f"line {line_number}: expected comma after label {name!r}")
        index += 1
    return labels


def _parse_quoted_label_value(source: str, index: int, line_number: int) -> tuple[str, int]:
    value: list[str] = []
    while index < len(source):
        char = source[index]
        if char == '"':
            return "".join(value), index + 1
        if char != "\\":
            value.append(char)
            index += 1
            continue
        index += 1
        if index >= len(source):
            break
        escaped = source[index]
        decoded = {"\\": "\\", '"': '"', "n": "\n"}.get(escaped)
        if decoded is None:
            raise MetricsParseError(
                f"line {line_number}: unsupported label escape \\{escaped}"
            )
        value.append(decoded)
        index += 1
    raise MetricsParseError(f"line {line_number}: unterminated quoted label value")


def _group_from_labels(labels: Labels, sample_name: str) -> GroupKey:
    label_dict = dict(labels)
    if set(label_dict) != _GROUP_LABELS:
        raise MetricsParseError(
            f"{sample_name} has labels {sorted(label_dict)!r}; expected exactly "
            "['engine', 'model_name'] for vLLM 0.27.0 timing histograms"
        )
    return label_dict["model_name"], label_dict["engine"]


def _format_group(group: GroupKey) -> str:
    return f"model_name={group[0]!r}, engine={group[1]!r}"


__all__ = [
    "TIMING_METRICS",
    "MetricsParseError",
    "fetch_metrics",
    "metric_total",
    "summarize_metrics_delta",
]
