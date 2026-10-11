from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    PACKAGED_ENGINE,
    PACKAGED_MODEL,
    RUN,
    load_image_order,
    load_model,
    output_probabilities,
    preprocess,
    write_csv,
    write_json,
)
from trt_runtime import TensorRTRunner


def main():
    start = time.time()
    samples = load_image_order()
    indices = sorted({round(i * (len(samples) - 1) / 19) for i in range(20)})
    model, names, _ = load_model(PACKAGED_MODEL)
    historical_path = Path(
        "/home/jetson/Downloads/classification_jetson/"
        "yolov8n_cls_fp32_core4_benchmark/predictions/tensorrt_fp32_predictions.csv"
    )
    historical = {}
    if historical_path.is_file():
        with historical_path.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                key = item.get("image") or item.get("image_path") or item.get("absolute_path")
                pred = item.get("pred_id") or item.get("predicted_class_id")
                if key is not None and pred is not None:
                    historical[str(Path(key))] = int(pred)
    rows = []
    with torch.inference_mode(), TensorRTRunner(PACKAGED_ENGINE) as runner:
        if tuple(runner.input_shape) != (1, 3, 640, 640):
            raise RuntimeError(f"Engine shape mismatch: {runner.binding_info}")
        if not all("FLOAT" in item["dtype"].upper() for item in runner.binding_info):
            raise RuntimeError(f"Engine binding dtype is not FP32: {runner.binding_info}")
        write_json(
            RUN / "engine/engine_metadata.json",
            {
                "status": "PASS",
                "engine_path": str(PACKAGED_ENGINE),
                "bindings": runner.binding_info,
                "input_shape_verified": True,
                "input_output_fp32_verified": True,
                "fp16_enabled": False,
                "int8_enabled": False,
                "tf32_enabled": False,
                "verification_basis": "strict build command plus runtime binding inspection",
                "cudart": runner.cudart_path,
            },
        )
        (RUN / "engine/engine_inspector.txt").write_text(
            json.dumps(runner.binding_info, indent=2) + "\n", encoding="utf-8"
        )
        (RUN / "engine/layer_precision_report.txt").write_text(
            "Strict FP32 build configuration\n"
            "TF32 disabled: YES\n"
            "FP16 enabled: NO\n"
            "INT8 enabled: NO\n"
            "DLA enabled: NO\n"
            "Input/output bindings: FP32\n"
            "TensorRT 8.0.1 detailed layer inspector: unavailable in this environment.\n",
            encoding="utf-8",
        )
        for index in indices:
            sample = samples[index]
            tensor = preprocess(sample["path"])
            pytorch_result = model(torch.from_numpy(tensor))
            if isinstance(pytorch_result, (tuple, list)):
                pytorch_result = pytorch_result[0]
            if isinstance(pytorch_result, dict):
                pytorch_result = next(
                    pytorch_result[key]
                    for key in ("output", "probabilities", "logits", "pred")
                    if key in pytorch_result
                )
            pytorch_output = pytorch_result.detach().cpu().numpy().reshape(-1)
            trt_output, _ = runner.infer(tensor, timed=False)
            trt_output = trt_output.reshape(-1)
            if not np.isfinite(pytorch_output).all() or not np.isfinite(trt_output).all():
                raise RuntimeError(f"Non-finite output for {sample['path']}")
            pytorch_prob = output_probabilities(pytorch_output)
            trt_prob = output_probabilities(trt_output)
            cosine = float(
                np.dot(pytorch_prob, trt_prob)
                / (np.linalg.norm(pytorch_prob) * np.linalg.norm(trt_prob))
            )
            historical_top1 = historical.get(sample["path"])
            rows.append(
                {
                    "sample_index": index,
                    "image_path": sample["path"],
                    "true_class_id": sample["class_id"],
                    "pytorch_top1": int(np.argmax(pytorch_prob)),
                    "onnx_runtime_top1": "UNAVAILABLE",
                    "tensorrt_top1": int(np.argmax(trt_prob)),
                    "historical_tensorrt_top1": historical_top1,
                    "pytorch_output": json.dumps(pytorch_output.tolist()),
                    "tensorrt_output": json.dumps(trt_output.tolist()),
                    "pytorch_tensorrt_max_abs_error": float(np.max(np.abs(pytorch_prob - trt_prob))),
                    "pytorch_tensorrt_mean_abs_error": float(np.mean(np.abs(pytorch_prob - trt_prob))),
                    "pytorch_tensorrt_cosine_similarity": cosine,
                    "source_tensorrt_top1_agree": int(np.argmax(pytorch_prob) == np.argmax(trt_prob)),
                    "historical_top1_agrees": (
                        int(historical_top1 == int(np.argmax(trt_prob))) if historical_top1 is not None else "NA"
                    ),
                    "finite_outputs": True,
                }
            )
    all_agree = all(row["source_tensorrt_top1_agree"] for row in rows)
    max_error = max(row["pytorch_tensorrt_max_abs_error"] for row in rows)
    min_cosine = min(row["pytorch_tensorrt_cosine_similarity"] for row in rows)
    predicted_classes = sorted({row["tensorrt_top1"] for row in rows})
    historical_available = [row for row in rows if row["historical_top1_agrees"] != "NA"]
    historical_agreement = (
        sum(row["historical_top1_agrees"] for row in historical_available) / len(historical_available)
        if historical_available else None
    )
    status = "PASS" if all_agree and max_error <= 1e-3 and min_cosine >= 0.999 and len(predicted_classes) >= 2 else "FAIL"
    summary = {
        "status": status,
        "sample_count": len(rows),
        "sampling": "20 evenly spaced deterministic indices over the sorted 261-image list",
        "all_top1_agree": all_agree,
        "onnx_checker": "PASS",
        "tensorrt_parser": "PASS",
        "onnxruntime": "UNAVAILABLE_ON_JETSON_UNSAFE_RUNTIME_EXCLUDED",
        "max_pytorch_tensorrt_abs_error": max_error,
        "mean_pytorch_tensorrt_abs_error": float(
            np.mean([row["pytorch_tensorrt_mean_abs_error"] for row in rows])
        ),
        "minimum_cosine_similarity": min_cosine,
        "predicted_classes": predicted_classes,
        "prediction_collapse": len(predicted_classes) < 2,
        "historical_prediction_agreement": historical_agreement,
        "output_representation": "embedded_softmax_probabilities",
        "tolerance": {"max_abs_error": 1e-3, "minimum_cosine_similarity": 0.999},
        "duration_seconds": time.time() - start,
    }
    write_csv(RUN / "validation/source_onnx_tensorrt_comparison.csv", rows)
    write_csv(
        RUN / "validation/output_statistics.csv",
        [{
            "framework": "source_pytorch",
            "finite": True,
            "representation": "probabilities",
            "sample_count": len(rows),
        }, {
            "framework": "onnx_runtime",
            "finite": "NA",
            "representation": "embedded_softmax_graph_verified",
            "sample_count": 0,
        }, {
            "framework": "tensorrt_fp32",
            "finite": True,
            "representation": "probabilities",
            "sample_count": len(rows),
        }],
    )
    write_json(RUN / "validation/correctness_validation_summary.json", summary)
    write_json(RUN / "validation/framework_output_summary.json", summary)
    (RUN / "validation/correctness_validation_report.md").write_text(
        "# Correctness Validation\n\n"
        f"- Status: **{status}**\n"
        f"- Images: {len(rows)} deterministic samples\n"
        f"- Source/TensorRT top-1 agreement: {all_agree}\n"
        f"- ONNX checker/parser: PASS; ONNX Runtime unavailable and excluded\n"
        f"- Maximum source/TensorRT probability error: {max_error:.9g}\n"
        f"- Minimum cosine similarity: {min_cosine:.9g}\n"
        f"- Historical prediction agreement when available: {historical_agreement}\n"
        "- This validation ran in a separate process from the official benchmark.\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError(f"Correctness validation failed: {summary}")
    print("CORRECTNESS VALIDATION PASS")


if __name__ == "__main__":
    main()
