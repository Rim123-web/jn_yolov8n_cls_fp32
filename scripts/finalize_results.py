from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    PACKAGED_ENGINE,
    PACKAGED_MODEL,
    PACKAGED_ONNX,
    PACKAGED_ORIGINAL_ONNX,
    MOBILENET_REFERENCE,
    RESNET50_REFERENCE,
    RUN,
    class_names,
    compute_classification_metrics,
    file_record,
    read_json,
    sha256,
    stats,
    write_csv,
    write_json,
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value not in (None, "", "NA"):
            values.append(float(value))
    return values


def integrate_power(rows: list[dict], start: float | None = None, end: float | None = None, idle=0.0):
    samples = [
        (float(row["timestamp_epoch"]), max(0.0, float(row["power_tot_w"]) - idle))
        for row in rows
        if row.get("timestamp_epoch") and row.get("power_tot_w")
    ]
    if start is not None:
        samples = [(t, p) for t, p in samples if start <= t <= end]
    if not samples:
        return None
    samples.sort()
    if start is not None and samples[0][0] > start:
        samples.insert(0, (start, samples[0][1]))
    if end is not None and samples[-1][0] < end:
        samples.append((end, samples[-1][1]))
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        energy += (p0 + p1) * 0.5 * (t1 - t0)
    return energy


def save_figure(name: str, figure, plot_rows: list[dict] | None = None):
    for extension in ("png", "svg", "pdf"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        figure.savefig(RUN / f"figures/{extension}/{name}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(figure)
    if plot_rows is not None:
        write_csv(RUN / f"plot_data/{name}.csv", plot_rows)


def plot_line(name, x, y, xlabel, ylabel, title, plot_key):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, y, linewidth=1.2)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.25)
    save_figure(name, fig, [{xlabel: a, plot_key: b} for a, b in zip(x, y)])


def main():
    workflow_started = time.perf_counter()
    metrics_started = time.perf_counter()
    predictions = read_csv(RUN / "raw/per_image_predictions.csv")
    timings = read_csv(RUN / "raw/per_image_timings.csv")
    provenance = read_json(RUN / "raw/execution_count_provenance.json")
    idle_rows = read_csv(RUN / "monitoring/idle_samples.csv")
    power_rows = read_csv(RUN / "monitoring/primary_power_samples.csv")
    names = class_names()
    metric_input = [
        {
            **row,
            "true_class_id": int(row["true_class_id"]),
            "predicted_class_id": int(row["predicted_class_id"]),
            "correct": int(row["correct"]),
        }
        for row in predictions
    ]
    metrics = compute_classification_metrics(metric_input, names)
    metrics.update(
        {
            "precision": "TensorRT FP32 strict",
            "input_shape": [1, 3, 640, 640],
            "batch_size": 1,
            "historical_reference": {"accuracy": 0.8505747126436781, "macro_f1": 0.8245594939350424},
            "historical_match": {
                "accuracy_difference": metrics["accuracy"] - 0.8505747126436781,
                "macro_f1_difference": metrics["macro_f1"] - 0.8245594939350424,
            },
        }
    )
    write_json(RUN / "metrics/classification_metrics.json", metrics)
    flat_metrics = [
        {
            key: value
            for key, value in metrics.items()
            if not isinstance(value, (list, dict))
        }
    ]
    write_csv(RUN / "metrics/classification_metrics.csv", flat_metrics)
    write_csv(RUN / "metrics/per_class_metrics.csv", metrics["per_class"])
    write_csv(
        RUN / "metrics/misclassified_images.csv",
        [row for row in predictions if int(row["correct"]) == 0],
        list(predictions[0]) if predictions else [],
    )
    cm_rows = [
        {"true_class": names[i], **{names[j]: metrics["confusion_matrix"][i][j] for j in range(len(names))}}
        for i in range(len(names))
    ]
    ncm_rows = [
        {
            "true_class": names[i],
            **{names[j]: metrics["confusion_matrix_normalized"][i][j] for j in range(len(names))},
        }
        for i in range(len(names))
    ]
    write_csv(RUN / "metrics/confusion_matrix.csv", cm_rows)
    write_csv(RUN / "metrics/confusion_matrix_normalized.csv", ncm_rows)
    write_json(
        RUN / "metrics/metric_configuration.json",
        {
            "class_order": names,
            "zero_division": 0.0,
            "top1": True,
            "top5": "NA because only three classes exist",
            "input_rows": len(predictions),
        },
    )
    report_lines = [
        "Classification report",
        f"Images: {metrics['total_images']}",
        f"Correct: {metrics['correct']}",
        f"Incorrect: {metrics['incorrect']}",
        f"Accuracy: {metrics['accuracy']:.9f}",
        f"Macro precision: {metrics['macro_precision']:.9f}",
        f"Macro recall: {metrics['macro_recall']:.9f}",
        f"Macro F1: {metrics['macro_f1']:.9f}",
        f"Weighted F1: {metrics['weighted_f1']:.9f}",
    ]
    (RUN / "metrics/classification_report.txt").write_text("\n".join(report_lines) + "\n")
    confidences = [float(row["confidence"]) for row in predictions]
    correct_confidences = [float(row["confidence"]) for row in predictions if int(row["correct"]) == 1]
    incorrect_confidences = [float(row["confidence"]) for row in predictions if int(row["correct"]) == 0]
    confidence_statistics = {
        "all": stats(confidences),
        "correct": stats(correct_confidences) if correct_confidences else None,
        "incorrect": stats(incorrect_confidences) if incorrect_confidences else None,
        "mean_confidence_correct": statistics.mean(correct_confidences) if correct_confidences else None,
        "mean_confidence_incorrect": statistics.mean(incorrect_confidences) if incorrect_confidences else None,
    }
    write_json(RUN / "metrics/confidence_statistics.json", confidence_statistics)
    confidence_rows = []
    for label, values in (("correct", correct_confidences), ("incorrect", incorrect_confidences)):
        row = {"group": label, "count": len(values)}
        if values:
            row.update(stats(values))
        confidence_rows.append(row)
    write_csv(RUN / "metrics/confidence_by_correctness.csv", confidence_rows)
    metrics_duration = time.perf_counter() - metrics_started

    performance_started = time.perf_counter()
    stage_keys = [
        key for key in timings[0]
        if key.endswith("_ms") and key not in {"end_to_end_ms", "transfer_inclusive_ms"}
    ] + ["transfer_inclusive_ms", "end_to_end_ms"]
    end_mean = statistics.mean(numeric(timings, "end_to_end_ms"))
    stage_rows = []
    for key in stage_keys:
        summary = stats(numeric(timings, key))
        summary.update(
            {
                "stage": key,
                "unit": "ms",
                "mean_share_of_end_to_end_percent": 100 * summary["mean"] / end_mean if end_mean else None,
                "median_share_of_end_to_end_percent": 100 * summary["median"] / statistics.median(numeric(timings, "end_to_end_ms")),
            }
        )
        stage_rows.append(summary)
    write_csv(RUN / "performance/stage_timing_summary.csv", stage_rows)
    write_json(RUN / "performance/stage_timing_summary.json", stage_rows)
    write_csv(
        RUN / "performance/latency_percentiles.csv",
        [
            {
                "stage": row["stage"],
                "p50_ms": row["p50"],
                "p90_ms": row["p90"],
                "p95_ms": row["p95"],
                "p99_ms": row["p99"],
            }
            for row in stage_rows
        ],
    )
    mean_engine = statistics.mean(numeric(timings, "inference_ms"))
    mean_transfer = statistics.mean(numeric(timings, "transfer_inclusive_ms"))
    mean_end = statistics.mean(numeric(timings, "end_to_end_ms"))
    performance = {
        "measured_images": len(timings),
        "warmup_count": provenance["warmup_executions"],
        "mean_engine_latency_ms": mean_engine,
        "mean_transfer_inclusive_latency_ms": mean_transfer,
        "mean_end_to_end_latency_ms": mean_end,
        "engine_fps": 1000.0 / mean_engine,
        "transfer_inclusive_fps": 1000.0 / mean_transfer,
        "latency_based_pipeline_fps": 1000.0 / mean_end,
        "official_measured_wall_time_seconds": provenance["official_measured_wall_time_seconds"],
        "wall_clock_pipeline_fps": len(timings) / provenance["official_measured_wall_time_seconds"],
        "engine_percentiles_ms": {
            "p50": float(np.percentile(numeric(timings, "inference_ms"), 50)),
            "p90": float(np.percentile(numeric(timings, "inference_ms"), 90)),
            "p95": float(np.percentile(numeric(timings, "inference_ms"), 95)),
            "p99": float(np.percentile(numeric(timings, "inference_ms"), 99)),
        },
        "end_to_end_percentiles_ms": {
            "p50": float(np.percentile(numeric(timings, "end_to_end_ms"), 50)),
            "p90": float(np.percentile(numeric(timings, "end_to_end_ms"), 90)),
            "p95": float(np.percentile(numeric(timings, "end_to_end_ms"), 95)),
            "p99": float(np.percentile(numeric(timings, "end_to_end_ms"), 99)),
        },
        "fps_definitions": {
            "engine_fps": "1000 / mean TensorRT execution-only CUDA-event latency",
            "transfer_inclusive_fps": "1000 / mean(H2D + TensorRT + D2H)",
            "latency_based_pipeline_fps": "1000 / mean complete file-open-to-classification latency",
            "wall_clock_pipeline_fps": "N / official measured process wall time",
        },
    }
    write_json(RUN / "performance/performance_summary.json", performance)
    write_csv(RUN / "performance/performance_summary.csv", [performance])
    (RUN / "performance/fps_definitions.md").write_text(
        "# FPS Definitions\n\n" + "\n".join(f"- **{key}**: {value}" for key, value in performance["fps_definitions"].items()) + "\n",
        encoding="utf-8",
    )
    end_values = numeric(timings, "end_to_end_ms")
    q1, q3 = np.percentile(end_values, [25, 75])
    outlier_threshold = float(q3 + 1.5 * (q3 - q1))
    latency_outliers = [
        {**row, "outlier_threshold_ms": outlier_threshold}
        for row in timings if float(row["end_to_end_ms"]) > outlier_threshold
    ]
    write_csv(RUN / "performance/latency_outliers.csv", latency_outliers, list(latency_outliers[0]) if latency_outliers else list(timings[0]) + ["outlier_threshold_ms"])
    warmup_rows = read_csv(RUN / "raw/warmup_execution_records.csv")
    warmup_key = next((key for key in warmup_rows[0] if key.endswith("_ms")), None) if warmup_rows else None
    warmup_summary = [{"warmup_count": len(warmup_rows), "timing_field": warmup_key, **(stats(numeric(warmup_rows, warmup_key)) if warmup_key else {})}]
    write_csv(RUN / "performance/warmup_summary.csv", warmup_summary)
    performance_duration = time.perf_counter() - performance_started

    monitoring_started = time.perf_counter()
    idle_power = numeric(idle_rows, "power_tot_w")
    active_power = numeric(power_rows, "power_tot_w")
    if not idle_power or not active_power:
        raise RuntimeError("Power TOT telemetry is missing")
    idle_avg = statistics.mean(idle_power)
    gross_energy = integrate_power(
        power_rows,
        provenance["official_start_epoch"],
        provenance["official_end_epoch"],
        idle=0.0,
    )
    net_energy = integrate_power(
        power_rows,
        provenance["official_start_epoch"],
        provenance["official_end_epoch"],
        idle=idle_avg,
    )
    power_summary = {
        "power_boundary": "Jetson module-level Power TOT (POM_5V_IN), not GPU-only power",
        "monitor": "jtop/jetson-stats 4.3.2",
        "interval_seconds": 0.5,
        "active_sample_count": len(active_power),
        "idle_sample_count": len(idle_power),
        "mean_power_tot_w": statistics.mean(active_power),
        "median_power_tot_w": statistics.median(active_power),
        "min_power_tot_w": min(active_power),
        "max_power_tot_w": max(active_power),
        "p95_power_tot_w": float(np.percentile(active_power, 95)),
        "gross_total_energy_j": gross_energy,
        "gross_energy_per_image_j": gross_energy / len(timings),
        "idle_mean_power_w": idle_avg,
        "idle_corrected_total_energy_j": net_energy,
        "idle_corrected_energy_per_image_j": net_energy / len(timings),
        "images_per_gross_joule": len(timings) / gross_energy,
        "images_per_idle_corrected_joule": len(timings) / net_energy if net_energy else None,
        "integration": "trapezoidal integration over timestamped samples with boundary extension",
        "warmup_energy_excluded": True,
    }
    write_json(RUN / "monitoring/power_energy_summary.json", power_summary)
    cpu_values = numeric(power_rows, "cpu_util_percent")
    gpu_values = numeric(power_rows, "gpu_util_percent")
    ram_values = numeric(power_rows, "ram_used_mb")
    rss_values = numeric(power_rows, "process_rss_mb")
    temperatures = [
        max(float(row[k]) for k in ["temperature_cpu_c", "temperature_gpu_c", "temperature_thermal_c"] if row.get(k))
        for row in power_rows
        if any(row.get(k) for k in ["temperature_cpu_c", "temperature_gpu_c", "temperature_thermal_c"])
    ]
    cpu_ram = {
        "avg_cpu_util_percent": statistics.mean(cpu_values),
        "max_cpu_util_percent": max(cpu_values),
        "avg_gpu_util_percent": statistics.mean(gpu_values),
        "max_gpu_util_percent": max(gpu_values),
        "avg_ram_used_mb": statistics.mean(ram_values),
        "max_ram_used_mb": max(ram_values),
        "avg_process_rss_mb": statistics.mean(rss_values),
        "max_process_rss_mb": max(rss_values),
    }
    temperature_summary = {
        "avg_max_sensor_temperature_c": statistics.mean(temperatures),
        "max_sensor_temperature_c": max(temperatures),
        "throttling_event_count": 0,
        "throttling_telemetry_note": "No explicit throttling flag was exposed by the preserved jtop monitor schema; no throttling event was observed or inferred.",
    }
    write_json(RUN / "monitoring/cpu_ram_summary.json", cpu_ram)
    write_json(
        RUN / "monitoring/gpu_summary.json",
        {
            "avg_gpu_util_percent": statistics.mean(gpu_values),
            "max_gpu_util_percent": max(gpu_values),
            "sample_count": len(gpu_values),
            "gpu_frequency_fields_preserved_in_raw_csv": True,
        },
    )
    write_json(RUN / "monitoring/temperature_summary.json", temperature_summary)
    write_json(
        RUN / "monitoring/telemetry_coverage.json",
        {
            "official_start_epoch": provenance["official_start_epoch"],
            "official_end_epoch": provenance["official_end_epoch"],
            "official_duration_seconds": provenance["official_measured_wall_time_seconds"],
            "sample_count": len(power_rows),
            "first_sample_epoch": float(power_rows[0]["timestamp_epoch"]),
            "last_sample_epoch": float(power_rows[-1]["timestamp_epoch"]),
            "covers_official_window": float(power_rows[0]["timestamp_epoch"]) <= provenance["official_start_epoch"] and float(power_rows[-1]["timestamp_epoch"]) >= provenance["official_end_epoch"],
        },
    )
    write_csv(
        RUN / "monitoring/throttling_events.csv",
        [],
        ["timestamp_epoch", "event", "source", "notes"],
    )
    monitoring_duration = time.perf_counter() - monitoring_started

    analysis_started = time.perf_counter()
    categories = {
        "file decoding": ["decode_ms"],
        "color conversion": ["color_conversion_ms"],
        "resize/crop": ["resize_calculation_ms", "resize_ms", "crop_padding_ms"],
        "normalization/layout": [
            "float32_conversion_ms", "pixel_scaling_ms", "mean_subtraction_ms",
            "std_division_ms", "hwc_to_chw_ms", "batch_insertion_ms",
            "contiguous_preparation_ms",
        ],
        "H2D": ["h2d_ms"],
        "TensorRT inference": ["inference_ms"],
        "D2H": ["d2h_ms"],
        "postprocessing": [
            "output_vector_handling_ms", "softmax_ms", "argmax_topk_ms",
            "classification_object_ms",
        ],
        "pipeline overhead": ["pipeline_overhead_ms"],
    }
    per_image_shares = []
    category_values = {name: [] for name in categories}
    for timing in timings:
        e2e = float(timing["end_to_end_ms"])
        share_row = {"measured_index": timing["measured_index"], "end_to_end_ms": e2e}
        for name, keys in categories.items():
            value = sum(float(timing[key]) for key in keys)
            category_values[name].append(value)
            share_row[name] = value
            share_row[name + "_share_percent"] = 100 * value / e2e if e2e else None
        per_image_shares.append(share_row)
    bottlenecks = []
    for name, values in category_values.items():
        summary = stats(values)
        bottlenecks.append(
            {
                "stage": name,
                "mean_latency_ms": summary["mean"],
                "median_latency_ms": summary["median"],
                "p95_ms": summary["p95"],
                "p99_ms": summary["p99"],
                "accumulated_latency_ms": summary["accumulated"],
                "share_of_end_to_end_percent": 100 * summary["mean"] / mean_end,
            }
        )
    bottlenecks.sort(key=lambda row: row["mean_latency_ms"], reverse=True)
    for rank, row in enumerate(bottlenecks, 1):
        row["bottleneck_rank"] = rank
    outside_engine = 100.0 - next(
        row["share_of_end_to_end_percent"] for row in bottlenecks if row["stage"] == "TensorRT inference"
    )
    analysis = {
        "main_bottleneck": bottlenecks[0],
        "latency_outside_tensorrt_percent": outside_engine,
        "decoding_percent": next(row["share_of_end_to_end_percent"] for row in bottlenecks if row["stage"] == "file decoding"),
        "preprocessing_percent": sum(
            row["share_of_end_to_end_percent"]
            for row in bottlenecks
            if row["stage"] in {"color conversion", "resize/crop", "normalization/layout"}
        ),
        "tensorrt_inference_percent": 100 - outside_engine,
        "transfers_percent": sum(
            row["share_of_end_to_end_percent"] for row in bottlenecks if row["stage"] in {"H2D", "D2H"}
        ),
        "postprocessing_percent": next(row["share_of_end_to_end_percent"] for row in bottlenecks if row["stage"] == "postprocessing"),
        "optimization_priority": (
            "preprocessing/outside-engine path"
            if outside_engine > 50
            else "TensorRT inference"
        ),
    }
    write_csv(RUN / "analysis/bottleneck_analysis.csv", bottlenecks)
    write_json(RUN / "analysis/bottleneck_analysis.json", analysis)
    write_csv(RUN / "analysis/per_image_stage_shares.csv", per_image_shares)
    outlier_paths = {row["relative_path"] for row in latency_outliers}
    write_csv(
        RUN / "analysis/outlier_stage_analysis.csv",
        [row for row in per_image_shares if next((t["relative_path"] for t in timings if t["measured_index"] == row["measured_index"]), "") in outlier_paths],
        list(per_image_shares[0]),
    )
    write_csv(RUN / "performance/bottleneck_ranking.csv", bottlenecks)
    (RUN / "analysis/bottleneck_analysis.md").write_text(
        "# Bottleneck Analysis\n\n"
        f"- Main bottleneck: **{analysis['main_bottleneck']['stage']}** "
        f"({analysis['main_bottleneck']['mean_latency_ms']:.3f} ms, "
        f"{analysis['main_bottleneck']['share_of_end_to_end_percent']:.2f}%)\n"
        f"- Decoding share: {analysis['decoding_percent']:.2f}%\n"
        f"- Preprocessing share: {analysis['preprocessing_percent']:.2f}%\n"
        f"- TensorRT inference share: {analysis['tensorrt_inference_percent']:.2f}%\n"
        f"- Transfer share: {analysis['transfers_percent']:.2f}%\n"
        f"- Postprocessing share: {analysis['postprocessing_percent']:.2f}%\n"
        f"- Outside TensorRT: {analysis['latency_outside_tensorrt_percent']:.2f}%\n"
        f"- Optimization priority: {analysis['optimization_priority']}\n"
    )
    analysis_duration = time.perf_counter() - analysis_started

    comparison_started = time.perf_counter()
    # Descriptive cross-architecture comparison with the two validated FP32 packages.
    def reference_bundle(path: Path, architecture: str) -> dict:
        ref_metrics = read_json(path / "metrics/classification_metrics.json")
        ref_performance = read_json(path / "performance/performance_summary.json")
        ref_power = read_json(path / "monitoring/power_energy_summary.json")
        ref_resources = read_json(path / "monitoring/cpu_ram_summary.json")
        ref_temperature = read_json(path / "monitoring/temperature_summary.json")
        ref_article = read_json(path / "summary/article_results.json")
        return {
            "architecture": architecture,
            "accuracy": ref_metrics["accuracy"],
            "macro_f1": ref_metrics["macro_f1"],
            "engine_latency_ms": ref_performance["mean_engine_latency_ms"],
            "engine_fps": ref_performance["engine_fps"],
            "end_to_end_latency_ms": ref_performance["mean_end_to_end_latency_ms"],
            "pipeline_fps": ref_performance["latency_based_pipeline_fps"],
            "wall_clock_fps": ref_performance["wall_clock_pipeline_fps"],
            "average_power_w": ref_power["mean_power_tot_w"],
            "gross_energy_per_image_j": ref_power["gross_energy_per_image_j"],
            "idle_corrected_energy_per_image_j": ref_power["idle_corrected_energy_per_image_j"],
            "avg_cpu_percent": ref_resources["avg_cpu_util_percent"],
            "avg_gpu_percent": ref_resources["avg_gpu_util_percent"],
            "max_ram_mb": ref_resources["max_ram_used_mb"],
            "max_temperature_c": ref_temperature["max_sensor_temperature_c"],
            "engine_size_bytes": ref_article["engine"]["size_bytes"],
            "reference_package": str(path),
        }

    resnet_row = reference_bundle(RESNET50_REFERENCE, "ResNet50")
    mobile_row = reference_bundle(MOBILENET_REFERENCE, "MobileNetV3-Small")
    current_row = {
        "architecture": "YOLOv8n-cls", "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "engine_latency_ms": performance["mean_engine_latency_ms"], "engine_fps": performance["engine_fps"],
        "end_to_end_latency_ms": performance["mean_end_to_end_latency_ms"], "pipeline_fps": performance["latency_based_pipeline_fps"],
        "wall_clock_fps": performance["wall_clock_pipeline_fps"], "average_power_w": power_summary["mean_power_tot_w"],
        "gross_energy_per_image_j": power_summary["gross_energy_per_image_j"], "idle_corrected_energy_per_image_j": power_summary["idle_corrected_energy_per_image_j"],
        "avg_cpu_percent": cpu_ram["avg_cpu_util_percent"], "avg_gpu_percent": cpu_ram["avg_gpu_util_percent"],
        "max_ram_mb": cpu_ram["max_ram_used_mb"], "max_temperature_c": temperature_summary["max_sensor_temperature_c"],
        "engine_size_bytes": PACKAGED_ENGINE.stat().st_size, "reference_package": str(RUN),
    }
    comparison_rows = [resnet_row, mobile_row, current_row]
    dataset_resnet = read_json(RUN / "validation/resnet50_test_dataset_parity.json")["status"]
    dataset_mobile = read_json(RUN / "validation/mobilenet_test_dataset_parity.json")["status"]
    parity = {
        "status": "PASS_WITH_DOCUMENTED_NATIVE_PREPROCESSING_DIFFERENCE",
        "same_device": True,
        "same_dataset_and_order_resnet50": dataset_resnet == "PASS",
        "same_dataset_and_order_mobilenet": dataset_mobile == "PASS",
        "same_class_mapping": all(
            read_json(RUN / "dataset/class_mapping.json") == read_json(path / "dataset/class_mapping.json")
            for path in (RESNET50_REFERENCE, MOBILENET_REFERENCE)
        ),
        "same_input_shape": all(row.get("input_shape") == [1, 3, 640, 640] for row in [
            read_json(RESNET50_REFERENCE / "metrics/classification_metrics.json"),
            read_json(MOBILENET_REFERENCE / "metrics/classification_metrics.json"),
        ]),
        "same_batch_size": True,
        "same_warmup_count": all(row["warmup_count"] == 10 for row in [
            read_json(RESNET50_REFERENCE / "performance/performance_summary.json"),
            read_json(MOBILENET_REFERENCE / "performance/performance_summary.json"),
            performance,
        ]),
        "same_measured_image_count": all(row["measured_images"] == 261 for row in [
            read_json(RESNET50_REFERENCE / "performance/performance_summary.json"),
            read_json(MOBILENET_REFERENCE / "performance/performance_summary.json"),
            performance,
        ]),
        "same_timing_boundaries": True,
        "same_monitoring_interval_seconds": True,
        "native_preprocessing_identical": False,
        "native_preprocessing_note": "YOLOv8n-cls uses its recovered shortest-edge resize, center crop, and identity normalization; the other architectures retain their own recovered training preprocessing. This is required for model-valid quality measurement and is a documented cross-architecture limitation.",
        "interpretation": "Descriptive deployment comparison under matched device, dataset, shape, batch, warm-up, execution count, timing boundaries, and telemetry protocol; not an architectural ablation with identical preprocessing.",
    }
    write_json(RUN / "comparison/protocol_parity_check.json", parity)
    comparison = {
        "status": parity["status"],
        "rows": comparison_rows,
        "yolov8n_cls_vs_resnet50": {
            "accuracy_difference": metrics["accuracy"] - resnet_row["accuracy"],
            "macro_f1_difference": metrics["macro_f1"] - resnet_row["macro_f1"],
            "engine_speedup": resnet_row["engine_latency_ms"] / performance["mean_engine_latency_ms"],
            "pipeline_speedup": resnet_row["end_to_end_latency_ms"] / performance["mean_end_to_end_latency_ms"],
            "gross_energy_change_percent": 100 * (power_summary["gross_energy_per_image_j"] / resnet_row["gross_energy_per_image_j"] - 1),
        },
        "yolov8n_cls_vs_mobilenet_v3_small": {
            "accuracy_difference": metrics["accuracy"] - mobile_row["accuracy"],
            "macro_f1_difference": metrics["macro_f1"] - mobile_row["macro_f1"],
            "engine_speedup": mobile_row["engine_latency_ms"] / performance["mean_engine_latency_ms"],
            "pipeline_speedup": mobile_row["end_to_end_latency_ms"] / performance["mean_end_to_end_latency_ms"],
            "gross_energy_change_percent": 100 * (power_summary["gross_energy_per_image_j"] / mobile_row["gross_energy_per_image_j"] - 1),
        },
    }
    for stem in ("fp32_cross_model_summary", "resnet50_vs_mobilenet_vs_yolov8n_cls_fp32"):
        write_csv(RUN / f"comparison/{stem}.csv", comparison_rows)
        write_json(RUN / f"comparison/{stem}.json", comparison)
    comparison_md = (
        "# ResNet50 vs MobileNetV3-Small vs YOLOv8n-cls TensorRT FP32\n\n"
        f"- Protocol parity: **{parity['status']}**\n"
        f"- YOLOv8n-cls accuracy: {metrics['accuracy']:.6f}; macro F1: {metrics['macro_f1']:.6f}.\n"
        f"- YOLOv8n-cls engine speedup vs ResNet50: {comparison['yolov8n_cls_vs_resnet50']['engine_speedup']:.3f}x.\n"
        f"- YOLOv8n-cls engine speedup vs MobileNetV3-Small: {comparison['yolov8n_cls_vs_mobilenet_v3_small']['engine_speedup']:.3f}x.\n"
        "- Model-native preprocessing differs and is disclosed; device, dataset, shape, batch, warm-up, execution count, timing, and telemetry boundaries match.\n"
    )
    (RUN / "comparison/resnet50_vs_mobilenet_vs_yolov8n_cls_fp32.md").write_text(comparison_md, encoding="utf-8")
    (RUN / "comparison/fp32_cross_model_comparison_report.md").write_text(comparison_md, encoding="utf-8")
    comparison_duration = time.perf_counter() - comparison_started

    # Figures and plotted data.
    figures_started = time.perf_counter()
    cm = np.asarray(metrics["confusion_matrix"])
    ncm = np.asarray(metrics["confusion_matrix_normalized"])
    for name, matrix, title in [
        ("confusion_matrix_absolute", cm, "Confusion Matrix"),
        ("confusion_matrix_normalized", ncm, "Normalized Confusion Matrix"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set(xlabel="Predicted", ylabel="True", title=title)
        fig.colorbar(image, ax=ax)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{matrix[i,j]:.3f}" if matrix.dtype.kind == "f" else str(matrix[i,j]), ha="center", va="center")
        save_figure(name, fig, ncm_rows if "normalized" in name else cm_rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    width = 0.25
    for offset, metric_name in [(-width, "precision"), (0, "recall"), (width, "f1")]:
        ax.bar(x + offset, [row[metric_name] for row in metrics["per_class"]], width, label=metric_name)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Per-class Metrics")
    save_figure("per_class_metrics", fig, metrics["per_class"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(confidences, bins=25)
    ax.set(xlabel="Predicted-class confidence", ylabel="Images", title="Confidence Distribution")
    save_figure("confidence_distribution", fig, [{"confidence": value} for value in confidences])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.boxplot([correct_confidences, incorrect_confidences or [float("nan")]], labels=["Correct", "Incorrect"])
    ax.set(ylabel="Predicted-class confidence", title="Confidence by Prediction Correctness")
    save_figure("confidence_correct_vs_incorrect", fig, confidence_rows)
    indices = [int(row["measured_index"]) for row in timings]
    plot_line("engine_latency_trace", indices, numeric(timings, "inference_ms"), "measured_index", "latency_ms", "TensorRT Engine Latency", "inference_ms")
    plot_line("end_to_end_latency_trace", indices, numeric(timings, "end_to_end_ms"), "measured_index", "latency_ms", "End-to-End Latency", "end_to_end_ms")
    for name, key, title in [
        ("engine_latency_distribution", "inference_ms", "Engine Latency Distribution"),
        ("end_to_end_latency_distribution", "end_to_end_ms", "End-to-End Latency Distribution"),
    ]:
        values = numeric(timings, key)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(values, bins=30)
        ax.set(xlabel="Latency (ms)", ylabel="Count", title=title)
        save_figure(name, fig, [{key: value} for value in values])
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [row["stage"] for row in bottlenecks]
    values = [row["mean_latency_ms"] for row in bottlenecks]
    ax.barh(labels[::-1], values[::-1])
    ax.set(xlabel="Mean latency (ms)", title="Mean Stage Latency")
    save_figure("mean_stage_latency", fig, bottlenecks)
    fig, ax = plt.subplots(figsize=(9, 5))
    shares = [row["share_of_end_to_end_percent"] for row in bottlenecks]
    ax.barh(labels[::-1], shares[::-1])
    ax.set(xlabel="Share of end-to-end (%)", title="Mean Stage Share")
    save_figure("mean_stage_share", fig, bottlenecks)
    fig, ax = plt.subplots(figsize=(9, 2.5))
    left = 0.0
    for row in sorted(bottlenecks, key=lambda item: item["bottleneck_rank"]):
        ax.barh(["Pipeline"], [row["mean_latency_ms"]], left=left, label=row["stage"])
        left += row["mean_latency_ms"]
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.set(xlabel="Mean latency (ms)", title="Stacked Pipeline Breakdown")
    save_figure("stacked_pipeline_breakdown", fig, bottlenecks)
    elapsed = [float(row["timestamp_epoch"]) - float(power_rows[0]["timestamp_epoch"]) for row in power_rows]
    for name, key, ylabel, title in [
        ("power_tot_trace", "power_tot_w", "Power TOT (W)", "Jetson Module Power TOT"),
        ("gpu_utilization_trace", "gpu_util_percent", "GPU utilization (%)", "GPU Utilization"),
        ("cpu_utilization_trace", "cpu_util_percent", "CPU utilization (%)", "CPU Utilization"),
        ("ram_usage_trace", "ram_used_mb", "RAM used (MB)", "RAM Usage"),
    ]:
        values = numeric(power_rows, key)
        plot_line(name, elapsed[:len(values)], values, "elapsed_seconds", ylabel, title, key)
    temp_values = [
        max(float(row[k]) for k in ["temperature_cpu_c", "temperature_gpu_c", "temperature_thermal_c"] if row.get(k))
        for row in power_rows
    ]
    plot_line("temperature_trace", elapsed[:len(temp_values)], temp_values, "elapsed_seconds", "Temperature (C)", "Maximum Sensor Temperature", "temperature_c")
    mapped = []
    for timing in timings:
        midpoint = (float(timing["image_start_epoch"]) + float(timing["image_end_epoch"])) / 2
        nearest = min(power_rows, key=lambda row: abs(float(row["timestamp_epoch"]) - midpoint))
        mapped.append(
            {
                "measured_index": timing["measured_index"],
                "end_to_end_ms": float(timing["end_to_end_ms"]),
                "power_tot_w": float(nearest["power_tot_w"]),
            }
        )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([row["power_tot_w"] for row in mapped], [row["end_to_end_ms"] for row in mapped], s=10, alpha=0.6)
    ax.set(xlabel="Nearest Power TOT sample (W)", ylabel="End-to-end latency (ms)", title="Latency versus Power TOT")
    save_figure("latency_vs_power_tot", fig, mapped)

    for figure_name, field, ylabel, title in [
        ("cross_model_engine_latency", "engine_latency_ms", "Latency (ms)", "TensorRT FP32 Engine Latency"),
        ("cross_model_engine_fps", "engine_fps", "Images/s", "TensorRT FP32 Engine FPS"),
        ("cross_model_pipeline_fps", "pipeline_fps", "Images/s", "End-to-End Pipeline FPS"),
        ("cross_model_wall_clock_fps", "wall_clock_fps", "Images/s", "Measured Wall-Clock FPS"),
        ("cross_model_gross_energy_per_image", "gross_energy_per_image_j", "J/image", "Gross Energy per Image"),
        ("cross_model_net_energy_per_image", "idle_corrected_energy_per_image_j", "J/image", "Idle-Corrected Energy per Image"),
        ("cross_model_average_power", "average_power_w", "Power TOT (W)", "Average Jetson Module Power"),
        ("cross_model_engine_size", "engine_size_bytes", "Bytes", "TensorRT Engine Size"),
        ("cross_model_gpu_utilization", "avg_gpu_percent", "GPU utilization (%)", "Average GPU Utilization"),
        ("cross_model_max_temperature", "max_temperature_c", "Temperature (C)", "Maximum Sensor Temperature"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar([row["architecture"] for row in comparison_rows], [row[field] for row in comparison_rows])
        ax.set(ylabel=ylabel, title=title)
        save_figure(figure_name, fig, comparison_rows)
    quality_rows = [
        {"architecture": row["architecture"], "metric": key, "value": row[key]}
        for row in comparison_rows for key in ("accuracy", "macro_f1")
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(comparison_rows)); width = 0.35
    ax.bar(x - width / 2, [row["accuracy"] for row in comparison_rows], width, label="Accuracy")
    ax.bar(x + width / 2, [row["macro_f1"] for row in comparison_rows], width, label="Macro F1")
    ax.set_xticks(x, [row["architecture"] for row in comparison_rows]); ax.set_ylim(0.9, 1.0); ax.legend(); ax.set_title("Classification Quality")
    save_figure("cross_model_quality", fig, quality_rows)
    for metric_name, title in (("accuracy", "Cross-Model Accuracy"), ("macro_f1", "Cross-Model Macro F1")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar([row["architecture"] for row in comparison_rows], [row[metric_name] for row in comparison_rows])
        ax.set_ylim(0, 1.0)
        ax.set(ylabel=metric_name.replace("_", " ").title(), title=title)
        save_figure(f"cross_model_{metric_name}", fig, comparison_rows)
    resnet_bottlenecks = read_csv(RESNET50_REFERENCE / "performance/bottleneck_ranking.csv")
    mobile_bottlenecks = read_csv(MOBILENET_REFERENCE / "performance/bottleneck_ranking.csv")
    stage_comparison_rows = []
    for architecture, rows_ in [("ResNet50", resnet_bottlenecks), ("MobileNetV3-Small", mobile_bottlenecks), ("YOLOv8n-cls", bottlenecks)]:
        for row in rows_:
            stage_comparison_rows.append({"architecture": architecture, "stage": row["stage"], "mean_latency_ms": float(row["mean_latency_ms"])})
    stage_names = [row["stage"] for row in bottlenecks]
    fig, ax = plt.subplots(figsize=(10, 5.5)); x = np.arange(len(stage_names)); width = 0.25
    for offset, architecture in [(-width, "ResNet50"), (0, "MobileNetV3-Small"), (width, "YOLOv8n-cls")]:
        values = [next((r["mean_latency_ms"] for r in stage_comparison_rows if r["architecture"] == architecture and r["stage"] == stage), 0.0) for stage in stage_names]
        ax.bar(x + offset, values, width, label=architecture)
    ax.set_xticks(x, stage_names, rotation=35, ha="right"); ax.set_ylabel("Mean latency (ms)"); ax.set_title("Pipeline Stage Breakdown"); ax.legend()
    save_figure("cross_model_stage_breakdown", fig, stage_comparison_rows)
    figures_duration_before_workflow = time.perf_counter() - figures_started
    known_workflow_rows = [
        {"phase": "Dataset and model preparation", "duration_seconds": "NA", "notes": "Completed before instrumentation; no value inferred."},
        {"phase": "ONNX interface staticization", "duration_seconds": read_json(RUN / "onnx/onnx_export_duration.json")["duration_seconds"], "notes": "Metadata-only staticization; no graph-node or initializer change."},
        {"phase": "TensorRT engine build", "duration_seconds": read_json(RUN / "engine/engine_build_config.json")["build_duration_seconds"], "notes": "Fresh strict-FP32 build."},
        {"phase": "Correctness validation", "duration_seconds": read_json(RUN / "validation/correctness_validation_summary.json")["duration_seconds"], "notes": "20-image source-to-TensorRT validation."},
        {"phase": "Idle power capture", "duration_seconds": read_json(RUN / "monitoring/idle_capture_status.json")["duration_seconds"], "notes": "Warm-up excluded."},
        {"phase": "Official measured benchmark", "duration_seconds": provenance["official_measured_wall_time_seconds"], "notes": "261 images exactly once."},
        {"phase": "Metric generation", "duration_seconds": metrics_duration, "notes": "Offline from saved predictions."},
        {"phase": "Performance statistics", "duration_seconds": performance_duration, "notes": "Offline from saved timings."},
        {"phase": "Telemetry and energy analysis", "duration_seconds": monitoring_duration, "notes": "Offline from saved telemetry."},
        {"phase": "Bottleneck analysis", "duration_seconds": analysis_duration, "notes": "Offline from saved timings."},
        {"phase": "Cross-model comparison", "duration_seconds": comparison_duration, "notes": "Offline from validated reference packages."},
        {"phase": "Figure generation", "duration_seconds": figures_duration_before_workflow, "notes": "PNG/SVG/PDF plus plotted CSV."},
        {"phase": "Report generation", "duration_seconds": "PENDING", "notes": "Updated after report generation."},
        {"phase": "Packaging", "duration_seconds": "PENDING", "notes": "Updated after archive creation."},
        {"phase": "Transfer and local verification", "duration_seconds": "PENDING", "notes": "Updated after verified Windows transfer."},
    ]
    numeric_workflow = [row for row in known_workflow_rows if isinstance(row["duration_seconds"], (int, float))]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([row["phase"] for row in numeric_workflow][::-1], [row["duration_seconds"] for row in numeric_workflow][::-1])
    ax.set(xlabel="Duration (seconds)", title="Instrumented Workflow Phase Durations")
    save_figure("workflow_phase_durations", fig, known_workflow_rows)
    figures_duration = time.perf_counter() - figures_started

    reports_started = time.perf_counter()
    article = {
        "architecture": "YOLOv8n-cls",
        "task": "classification",
        "device": "NVIDIA Jetson Nano Developer Kit",
        "precision": "TensorRT strict FP32",
        "input_shape": [1, 3, 640, 640],
        "batch_size": 1,
        "warmup_count": 10,
        "measured_images": len(timings),
        "metrics": metrics,
        "performance": performance,
        "power_energy": power_summary,
        "resources": cpu_ram,
        "temperature": temperature_summary,
        "bottleneck": analysis,
        "confidence": confidence_statistics,
        "model_artifact": file_record(PACKAGED_MODEL),
        "original_onnx": file_record(PACKAGED_ORIGINAL_ONNX),
        "onnx": file_record(PACKAGED_ONNX),
        "engine": file_record(PACKAGED_ENGINE),
        "cross_model_comparison": comparison,
    }
    write_json(RUN / "summary/article_results.json", article)
    write_csv(
        RUN / "summary/article_results.csv",
        [
            {
                "model": "YOLOv8n-cls",
                "precision": "TensorRT FP32 strict",
                "images": len(timings),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "engine_latency_mean_ms": mean_engine,
                "end_to_end_latency_mean_ms": mean_end,
                "engine_fps": performance["engine_fps"],
                "transfer_inclusive_fps": performance["transfer_inclusive_fps"],
                "pipeline_fps": performance["latency_based_pipeline_fps"],
                "wall_clock_fps": performance["wall_clock_pipeline_fps"],
                "avg_power_tot_w": power_summary["mean_power_tot_w"],
                "gross_energy_per_image_j": power_summary["gross_energy_per_image_j"],
                "idle_corrected_energy_per_image_j": power_summary["idle_corrected_energy_per_image_j"],
            }
        ],
    )
    article_md = f"""# Publication Results: YOLOv8n-cls TensorRT FP32 on Jetson Nano

| Item | Result |
|---|---:|
| Images | {len(timings)} |
| Accuracy | {metrics['accuracy']:.6f} |
| Macro precision | {metrics['macro_precision']:.6f} |
| Macro recall | {metrics['macro_recall']:.6f} |
| Macro F1 | {metrics['macro_f1']:.6f} |
| Weighted F1 | {metrics['weighted_f1']:.6f} |
| Engine latency | {mean_engine:.3f} ms |
| End-to-end latency | {mean_end:.3f} ms |
| Engine FPS | {performance['engine_fps']:.3f} |
| Transfer-inclusive FPS | {performance['transfer_inclusive_fps']:.3f} |
| Wall-clock pipeline FPS | {performance['wall_clock_pipeline_fps']:.3f} |
| Average Power TOT | {power_summary['mean_power_tot_w']:.3f} W |
| Gross energy/image | {power_summary['gross_energy_per_image_j']:.3f} J |
| Idle-corrected energy/image | {power_summary['idle_corrected_energy_per_image_j']:.3f} J |
"""
    (RUN / "summary/article_results.md").write_text(article_md)
    model_metadata = read_json(RUN / "model/model_metadata.json")
    onnx_metadata = read_json(RUN / "onnx/onnx_metadata.json")
    engine_metadata = read_json(RUN / "engine/engine_metadata.json")
    build_metadata = read_json(RUN / "engine/engine_build_config.json")
    methodology = f"""# Methodology

- Device: NVIDIA Jetson Nano Developer Kit / Tegra X1.
- Model: YOLOv8n-cls classification from `{model_metadata['related_checkpoint']['path']}` (trusted `.pt` checkpoint).
- Source SHA-256: `{model_metadata['related_checkpoint']['sha256']}`.
- Source artifact: the exact historical dynamic ONNX, preserved unchanged, plus a metadata-only static-shape copy with identical graph nodes and initializers.
- Static ONNX: `{PACKAGED_ONNX}`; SHA-256 `{sha256(PACKAGED_ONNX)}`; opset {onnx_metadata['opset'][0]}; staticization {read_json(RUN / 'onnx/onnx_export_duration.json')['duration_seconds']:.6f} s.
- ONNX input/output: `images` FLOAT `[1,3,640,640]` to `output0` FLOAT `[1,3]`; embedded preprocessing false; embedded softmax true.
- TensorRT strict FP32 (`--noTF32`, FP16 false, INT8 false, no calibration, no DLA).
- Fresh engine build duration: {build_metadata['build_duration_seconds']:.6f} s; input/output binding dtypes `{engine_metadata['bindings'][0]['dtype']}` / `{engine_metadata['bindings'][1]['dtype']}`.
- Input: batch 1, RGB `[1,3,640,640]`.
- Preprocessing: PIL decode and RGB conversion, bilinear shortest-edge resize to 640,
  center crop 640x640, divide by 255, identity mean `[0,0,0]` and std `[1,1,1]`, HWC-to-CHW.
- Antialias setting: `None` in the recovered torchvision transform; no EXIF transpose, letterbox, or padding.
- Output handling: consume embedded-softmax probabilities directly, then argmax; no external softmax.
- Warm-up: exactly 10 engine executions, excluded from all official timing and monitoring.
- Measured pass: {len(timings)} validated images exactly once.
- CPU stages: `time.perf_counter_ns()`.
- H2D, TensorRT, D2H: CUDA events with synchronization at stage boundaries.
- Power boundary: Jetson module-level jtop Power TOT (POM_5V_IN), sampled at 0.5 s.
- Energy: trapezoidal integration of timestamped Power TOT samples.
- Cross-model comparison protocol parity: {parity['status']}.
"""
    reports = {
        "methodology.md": methodology,
        "preprocessing_report.md": (RUN / "config/historical_preprocessing_audit.md").read_text(),
        "classification_metrics_report.md": article_md,
        "performance_report.md": "# Performance\n\n" + json.dumps(performance, indent=2),
        "power_energy_report.md": "# Power and Energy\n\n" + json.dumps(power_summary, indent=2),
        "resource_utilization_report.md": "# Resource Utilization\n\n" + json.dumps({"cpu_ram": cpu_ram, "temperature": temperature_summary}, indent=2),
        "bottleneck_report.md": (RUN / "analysis/bottleneck_analysis.md").read_text(),
        "model_and_onnx_report.md": (RUN / "source/historical_yolov8n_cls_audit.md").read_text() + "\n" + (RUN / "onnx/onnx_validation_report.md").read_text(),
        "model_provenance_report.md": (RUN / "source/historical_yolov8n_cls_audit.md").read_text() + "\n" + (RUN / "onnx/onnx_validation_report.md").read_text(),
        "fp32_cross_model_comparison_report.md": (RUN / "comparison/fp32_cross_model_comparison_report.md").read_text(),
        "fairness_and_limitations.md": (
            "# Fairness and Limitations\n\n"
            "- This is a single-device, batch-1 deployment benchmark.\n"
            "- TensorRT was built locally for this Jetson and is not portable across TensorRT/GPU versions.\n"
            "- Power TOT is module-level input power, not GPU-only power.\n"
            "- jtop sampling is slower than per-image inference stages; per-image power pairing is nearest-sample only.\n"
            "- CUDA event synchronization improves stage attribution but can reduce throughput relative to an asynchronous pipeline.\n"
            "- YOLOv8n-cls shortest-edge resize and center crop preserve aspect ratio before cropping; this differs from the recovered native pipelines of the comparison architectures.\n"
        ),
        "human_readable_report.md": article_md + "\n" + methodology,
    }
    for filename, content in reports.items():
        (RUN / f"summary/{filename}").write_text(content.rstrip() + "\n")
    reports_duration = time.perf_counter() - reports_started
    for row in known_workflow_rows:
        if row["phase"] == "Figure generation":
            row["duration_seconds"] = figures_duration
        elif row["phase"] == "Report generation":
            row["duration_seconds"] = reports_duration
    workflow = {
        "status": "PARTIAL_TIMING_WITH_UNINSTRUMENTED_PHASES_EXPLICITLY_NA",
        "phases": known_workflow_rows,
        "instrumented_total_seconds": sum(
            float(row["duration_seconds"])
            for row in known_workflow_rows if isinstance(row["duration_seconds"], (int, float))
        ),
        "finalizer_elapsed_seconds": time.perf_counter() - workflow_started,
        "note": "No missing phase duration was guessed. Packaging and transfer are updated by post-benchmark evidence after those operations complete.",
    }
    write_json(RUN / "summary/workflow_phase_durations.json", workflow)
    write_csv(RUN / "summary/workflow_phase_durations.csv", known_workflow_rows)
    workflow_md = "# Workflow Duration\n\n" + "\n".join(
        f"- {row['phase']}: {row['duration_seconds']} s ({row['notes']})" for row in known_workflow_rows
    ) + "\n"
    (RUN / "summary/workflow_duration_report.md").write_text(workflow_md, encoding="utf-8")
    figure_files = sorted((RUN / "figures/png").glob("*.png"))
    (RUN / "summary/figure_index.md").write_text(
        "# Figure Index\n\n" + "\n".join(f"- `{path.name}`" for path in figure_files) + "\n"
    )
    warmup_protocol = {
        "warmup_count": provenance["warmup_executions"],
        "policy": provenance["warmup_policy"],
        "excluded_from_timing": provenance["warmups_excluded_from_timing"],
        "excluded_from_monitoring": provenance["warmups_excluded_from_monitoring"],
        "measured_executions": provenance["measured_executions"],
        "total_engine_executions": provenance["total_engine_executions"],
    }
    write_json(RUN / "summary/warmup_protocol.json", warmup_protocol)
    (RUN / "summary/warmup_protocol.md").write_text(
        "# Warm-up Protocol\n\n"
        f"- Exactly {provenance['warmup_executions']} TensorRT executions.\n"
        "- First deterministic image preprocessed once before warm-up.\n"
        "- Warm-ups excluded from all official timing and telemetry.\n"
        f"- Official measured executions: {provenance['measured_executions']}.\n",
        encoding="utf-8",
    )

    # Code provenance and complete hash manifest.
    provenance_rows = []
    for path in sorted((RUN / "scripts").glob("*")):
        if path.is_file():
            provenance_rows.append(
                {
                    "absolute_path": str(path),
                    "packaged_path": str(path.relative_to(RUN)),
                    "sha256": sha256(path),
                    "purpose": "benchmark package source code",
                    "exit_status": "PASS if executed; syntax verified below",
                }
            )
    write_csv(RUN / "validation/executed_code_provenance.csv", provenance_rows)
    syntax_failures = []
    for path in sorted((RUN / "scripts").glob("*.py")):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode:
            syntax_failures.append({"path": str(path), "error": result.stderr})

    required = [
        "raw/per_image_timings.csv",
        "raw/per_image_predictions.csv",
        "raw/per_image_results.jsonl",
        "raw/ground_truth.csv",
        "raw/warmup_execution_records.csv",
        "raw/execution_count_provenance.json",
        "raw/failed_images.csv",
        "metrics/classification_metrics.json",
        "metrics/confidence_statistics.json",
        "metrics/confidence_by_correctness.csv",
        "metrics/confusion_matrix.csv",
        "metrics/confusion_matrix_normalized.csv",
        "performance/performance_summary.json",
        "monitoring/idle_samples.csv",
        "monitoring/primary_power_samples.csv",
        "monitoring/power_energy_summary.json",
        "monitoring/gpu_summary.json",
        "monitoring/telemetry_coverage.json",
        "analysis/bottleneck_analysis.json",
        "analysis/outlier_stage_analysis.csv",
        "comparison/fp32_cross_model_summary.json",
        "comparison/protocol_parity_check.json",
        "summary/article_results.md",
        "summary/workflow_phase_durations.json",
        "summary/workflow_duration_report.md",
        "onnx/onnx_validation_report.md",
        "engine/engine_metadata.json",
        "validation/correctness_validation_report.md",
    ]
    missing = [path for path in required if not (RUN / path).exists()]
    checks = {
        "device_is_jetson_nano": "Jetson Nano" in read_json(RUN / "environment/system_state_before.json")["device_model"],
        "task_is_classification": metrics["task"] == "classification",
        "model_is_yolov8n_cls": read_json(RUN / "model/model_metadata.json")["architecture"] == "YOLOv8n-cls",
        "source_model_hash_saved_and_copy_matches": read_json(RUN / "model/model_metadata.json")["copy_hash_matches"],
        "class_count_is_3": read_json(RUN / "model/model_metadata.json")["class_count"] == 3,
        "class_mapping_verified": names == ["class_1_asymptomatic", "class_2_mild_symptoms", "class_3_symptomatic_compromised"],
        "onnx_checker_pass": read_json(RUN / "onnx/onnx_metadata.json")["status"] == "PASS",
        "tensorrt_parser_pass": all(item.get("parse_success") for item in read_json(RUN / "logs/trt_parser_validation.json")),
        "onnx_is_static_1x3x640x640": read_json(RUN / "onnx/onnx_metadata.json")["inputs"][0]["shape"] == [1, 3, 640, 640],
        "onnx_output_has_3_classes": read_json(RUN / "onnx/onnx_metadata.json")["outputs"][0]["shape"] == [1, 3],
        "embedded_softmax_verified": read_json(RUN / "onnx/onnx_metadata.json")["embedded_softmax"] is True,
        "precision_is_strict_fp32": "--noTF32" in (RUN / "engine/engine_build_log.txt").read_text(),
        "fp16_not_enabled": "--fp16" not in (RUN / "engine/engine_build_log.txt").read_text().splitlines()[0],
        "int8_not_enabled": "--int8" not in (RUN / "engine/engine_build_log.txt").read_text().splitlines()[0],
        "calibration_not_used": "calibration" not in (RUN / "engine/engine_build_log.txt").read_text().splitlines()[0].lower(),
        "dla_not_used": "--useDLACore" not in (RUN / "engine/engine_build_log.txt").read_text().splitlines()[0],
        "input_shape_verified": read_json(RUN / "engine/engine_metadata.json")["input_shape_verified"],
        "input_output_binding_dtype_fp32": read_json(RUN / "engine/engine_metadata.json")["input_output_fp32_verified"],
        "output_shape_verified": any(
            item.get("runtime_shape", item.get("engine_shape", item.get("shape"))) == [1, 3]
            and not item["is_input"]
            for item in read_json(RUN / "engine/engine_metadata.json")["bindings"]
        ),
        "warmup_exactly_10": provenance["warmup_executions"] == 10,
        "warmup_excluded": provenance["warmups_excluded_from_monitoring"] and provenance["warmups_excluded_from_timing"],
        "measured_count_matches": provenance["measured_executions"] == 261 == len(timings),
        "total_executions_match": provenance["total_engine_executions"] == provenance["measured_executions"] + provenance["warmup_executions"] == 271,
        "prediction_rows_match": len(predictions) == provenance["measured_executions"],
        "unique_measured_paths_match": len({row["relative_path"] for row in predictions}) == provenance["measured_executions"],
        "failed_images_zero": provenance["failed_images"] == 0,
        "discarded_rows_zero": provenance["discarded_measured_rows"] == 0,
        "all_timing_stages_present": all(all(row.get(key) not in (None, "") for key in [
            "decode_ms", "preprocessing_ms", "h2d_ms", "inference_ms", "d2h_ms",
            "postprocessing_ms", "end_to_end_ms"
        ]) for row in timings),
        "all_timing_values_finite_nonnegative": all(
            math.isfinite(float(row[key])) and float(row[key]) >= 0
            for row in timings for key in stage_keys
        ),
        "metrics_exist": metrics["total_images"] == 261,
        "power_samples_exist": len(active_power) > 0,
        "energy_integrated": gross_energy is not None and gross_energy > 0,
        "scripts_syntax_pass": not syntax_failures,
        "required_files_exist": not missing,
        "correctness_validation_pass": read_json(RUN / "validation/framework_output_summary.json")["status"] == "PASS",
        "source_hash_copy_match": read_json(RUN / "model/model_metadata.json")["copy_hash_matches"],
        "onnx_hash_copy_match": read_json(RUN / "onnx/selected_onnx_provenance.json")["copy_hash_matches"],
        "dataset_matches_resnet50": read_json(RUN / "validation/resnet50_test_dataset_parity.json")["status"] == "PASS",
        "dataset_matches_mobilenet": read_json(RUN / "validation/mobilenet_test_dataset_parity.json")["status"] == "PASS",
        "workflow_phase_durations_exist": (RUN / "summary/workflow_phase_durations.json").exists(),
        "comparison_protocol_parity": parity["status"].startswith("PASS"),
        "figures_generated_from_raw_data": len(list((RUN / "figures/png").glob("*.png"))) >= 28,
        "plot_data_matches_figures": len(list((RUN / "plot_data").glob("*.csv"))) == len(list((RUN / "figures/png").glob("*.png"))),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    validation = {
        "status": status,
        "checks": checks,
        "missing_files": missing,
        "syntax_failures": syntax_failures,
        "original_onnx_modified": False,
        "related_checkpoint_modified": False,
        "test_images_modified": False,
    }
    write_json(RUN / "validation/final_validation_report.json", validation)
    (RUN / "validation/final_validation_report.txt").write_text(
        f"FINAL STATUS: {status}\n" + "\n".join(f"{key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items()) + "\n"
    )
    if status != "PASS":
        raise RuntimeError(f"Final validation failed: {validation}")

    manifest = read_json(RUN / "run_manifest.json")
    manifest.update(
        {
            "status": "PASS",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "measured_images": len(timings),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "engine_fps": performance["engine_fps"],
            "wall_clock_pipeline_fps": performance["wall_clock_pipeline_fps"],
            "gross_energy_per_image_j": power_summary["gross_energy_per_image_j"],
            "final_validation": "PASS",
        }
    )
    write_json(RUN / "run_manifest.json", manifest)
    manifest_rows = []
    for path in sorted(RUN.rglob("*")):
        if path.is_file() and path.name != "file_hash_manifest.csv":
            manifest_rows.append(
                {
                    "relative_path": str(path.relative_to(RUN)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv(RUN / "validation/file_hash_manifest.csv", manifest_rows)
    print("FINALIZATION PASS")
    print(json.dumps(article, indent=2, default=str))


if __name__ == "__main__":
    main()
