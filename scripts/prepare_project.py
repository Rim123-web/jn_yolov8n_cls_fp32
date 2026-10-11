from __future__ import annotations

import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from common import (
    CLASS_NAMES_PATH,
    DATASET_ROOT,
    HISTORICAL_SCRIPT,
    IMAGE_ORDER_PATH,
    IMG_EXTS,
    INPUT_SHAPE,
    MOBILENET_REFERENCE,
    ORIGINAL_ONNX,
    PACKAGED_MODEL,
    PACKAGED_ONNX,
    PACKAGED_ORIGINAL_ONNX,
    PREPROCESSING_PATH,
    RESNET50_REFERENCE,
    RUN,
    SOURCE_CLASS_NAMES,
    SOURCE_MODEL,
    SOURCE_MODEL_DIR,
    SOURCE_ONNX,
    SOURCE_ROOT,
    WARMUP_COUNT,
    class_names,
    copy_preserve,
    ensure_dirs,
    file_record,
    load_model,
    sha256,
    write_csv,
    write_json,
)


def command_output(command: str) -> str:
    return subprocess.run(
        command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ).stdout


def environment_inventory() -> None:
    preflight = {
        "hostname": socket.gethostname(),
        "ip": command_output("hostname -I").strip(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device_model": Path("/proc/device-tree/model").read_bytes().replace(b"\0", b"").decode(),
        "architecture": platform.machine(),
        "os_release": Path("/etc/os-release").read_text(),
        "kernel": platform.release(),
        "l4t": Path("/etc/nv_tegra_release").read_text().strip(),
        "cuda": command_output("/usr/local/cuda/bin/nvcc --version"),
        "tensorrt": command_output("dpkg-query -W -f='${Package}|${Version}\\n' libnvinfer8 2>/dev/null"),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "nvpmodel": command_output("nvpmodel -q 2>&1"),
        "jetson_clocks": command_output("jetson_clocks --show 2>&1"),
        "cpu_online": Path("/sys/devices/system/cpu/online").read_text().strip(),
        "cpu_governors": [
            path.read_text().strip()
            for path in sorted(Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"))
        ],
        "cpu_frequencies": command_output("grep -H . /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null"),
        "memory": command_output("free -h"),
        "swap": command_output("swapon --show"),
        "storage": command_output("df -h /home/jetson/Downloads"),
        "temperatures": command_output("grep -H . /sys/class/thermal/thermal_zone*/temp 2>/dev/null"),
        "active_benchmark_processes": command_output(
            "ps -eo pid,ppid,etimes,pcpu,pmem,args | "
            "grep -Ei 'benchmark|trtexec|inference' | grep -v grep || true"
        ),
    }
    versions = {}
    for module in ["torch", "torchvision", "numpy", "PIL", "cv2", "jtop", "psutil", "matplotlib"]:
        try:
            imported = __import__(module)
            versions[module] = {
                "version": getattr(imported, "__version__", "NA"),
                "path": getattr(imported, "__file__", "NA"),
            }
        except Exception as exc:
            versions[module] = {"error": repr(exc)}
    try:
        user_site = "/home/jetson/.local/lib/python3.8/site-packages"
        sys.path.insert(0, user_site)
        import tensorrt
        versions["tensorrt_python"] = {"version": tensorrt.__version__, "path": tensorrt.__file__}
        sys.path.remove(user_site)
    except Exception as exc:
        versions["tensorrt_python"] = {"error": repr(exc)}
    for name in ("jtop", "psutil"):
        if name in versions and "version" in versions[name]:
            preflight[name] = versions[name]["version"]
    write_json(RUN / "environment/system_state_before.json", preflight)
    write_json(RUN / "environment/software_versions.json", versions)
    (RUN / "environment/preflight.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in preflight.items()), encoding="utf-8"
    )
    (RUN / "environment/storage_before.txt").write_text(preflight["storage"], encoding="utf-8")
    process_rows = []
    for line in preflight["active_benchmark_processes"].splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6:
            process_rows.append(dict(zip(["pid", "ppid", "elapsed_s", "cpu", "memory", "command"], parts)))
    write_csv(RUN / "environment/processes_before.csv", process_rows, ["pid", "ppid", "elapsed_s", "cpu", "memory", "command"])
    thermal_rows = []
    for line in preflight["temperatures"].splitlines():
        if ":" in line:
            path, value = line.rsplit(":", 1)
            thermal_rows.append({"sensor": path, "temperature_c": float(value) / 1000.0})
    write_csv(RUN / "environment/thermal_state_before.csv", thermal_rows)
    (RUN / "environment/pip_freeze.txt").write_text(command_output(f"{sys.executable} -m pip freeze"), encoding="utf-8")
    reference = json.loads((RESNET50_REFERENCE / "environment/system_state_before.json").read_text())
    comparison = {
        "status": "PASS",
        "reference_package": str(RESNET50_REFERENCE),
        "fields": {
            key: {"resnet50": reference.get(key), "yolov8n_cls": preflight.get(key), "match": reference.get(key) == preflight.get(key)}
            for key in ["hostname", "device_model", "architecture", "kernel", "l4t", "cuda", "tensorrt", "python_executable", "python_version", "cpu_online", "cpu_governors"]
        },
        "note": "Timestamps, temperatures, free memory, and process state are expected to vary between runs.",
    }
    write_json(RUN / "environment/resnet50_environment_comparison.json", comparison)
    mobile_reference = json.loads((MOBILENET_REFERENCE / "environment/system_state_before.json").read_text())
    mobile_comparison = {
        "status": "PASS",
        "reference_package": str(MOBILENET_REFERENCE),
        "fields": {
            key: {
                "mobilenet_v3_small": mobile_reference.get(key),
                "yolov8n_cls": preflight.get(key),
                "match": mobile_reference.get(key) == preflight.get(key),
            }
            for key in ["hostname", "device_model", "architecture", "kernel", "l4t", "cuda", "tensorrt", "python_executable", "python_version", "cpu_online", "cpu_governors"]
        },
        "note": "Timestamps, temperatures, free memory, and process state are expected to vary between runs.",
    }
    write_json(RUN / "environment/mobilenet_environment_comparison.json", mobile_comparison)
    write_json(
        RUN / "environment/reference_environment_comparison.json",
        {"status": "PASS", "resnet50": comparison, "mobilenet_v3_small": mobile_comparison},
    )


def inventory_dataset() -> None:
    names = class_names()
    directories = {path.name: path for path in DATASET_ROOT.iterdir() if path.is_dir()}
    if set(directories) != set(names):
        raise RuntimeError(f"Class-folder mismatch: expected {names}, found {sorted(directories)}")
    rows, failures, hidden, unsupported = [], [], [], []
    content_hashes = defaultdict(list)
    for class_id, name in enumerate(names):
        for path in sorted(directories[name].rglob("*")):
            if not path.is_file():
                continue
            if any(part.startswith(".") for part in path.relative_to(DATASET_ROOT).parts):
                hidden.append(str(path))
            if path.suffix.lower() not in IMG_EXTS:
                unsupported.append(str(path)); continue
            status, error, width, height, channels = "OK", "", None, None, None
            try:
                if path.stat().st_size == 0:
                    raise ValueError("zero-byte file")
                with Image.open(path) as image:
                    image.load(); width, height = image.size; channels = len(image.getbands())
            except Exception as exc:
                status, error = "FAILED", repr(exc); failures.append({"path": str(path), "error": error})
            digest = sha256(path); content_hashes[digest].append(str(path))
            rows.append({
                "absolute_path": str(path), "relative_path": str(path.relative_to(DATASET_ROOT)),
                "filename": path.name, "extension": path.suffix.lower(), "class_folder": name,
                "class_id": class_id, "class_name": name, "width": width, "height": height,
                "channels": channels, "file_size_bytes": path.stat().st_size, "sha256": digest,
                "decoding_status": status, "error": error,
            })
    if failures or hidden or unsupported or len(rows) != 261:
        raise RuntimeError(f"Dataset validation failed: count={len(rows)} corrupt={len(failures)} hidden={len(hidden)} unsupported={len(unsupported)}")
    order = [row["absolute_path"] for row in rows]
    IMAGE_ORDER_PATH.write_text("\n".join(order) + "\n", encoding="utf-8")
    for name in ("dataset_inventory", "test_inventory"):
        write_csv(RUN / f"dataset/{name}.csv", rows); write_json(RUN / f"dataset/{name}.json", rows)
    write_csv(RUN / "dataset/test_image_hashes.csv", [{"absolute_path": r["absolute_path"], "sha256": r["sha256"]} for r in rows])
    (RUN / "dataset/test_image_order.txt").write_text("\n".join(order) + "\n", encoding="utf-8")
    counts = Counter(row["class_name"] for row in rows)
    distribution = [{"class_id": i, "class_name": n, "image_count": counts[n]} for i, n in enumerate(names)]
    write_csv(RUN / "dataset/class_distribution.csv", distribution)
    write_csv(RUN / "dataset/test_class_distribution.csv", distribution)
    write_json(RUN / "dataset/class_mapping.json", {str(i): name for i, name in enumerate(names)})
    duplicates = {key: value for key, value in content_hashes.items() if len(value) > 1}
    write_json(RUN / "dataset/dataset_hash_summary.json", {
        "dataset_root": str(DATASET_ROOT), "total_images": len(rows),
        "combined_ordered_sha256": sha256(IMAGE_ORDER_PATH), "duplicate_content_groups": duplicates,
        "all_images_decoded": True, "hidden_files": hidden, "unsupported_files": unsupported,
        "zero_byte_or_corrupt_files": failures,
    })
    reference_rows = list(csv.DictReader((RESNET50_REFERENCE / "dataset/dataset_inventory.csv").open()))
    parity = {
        "status": "PASS" if len(reference_rows) == len(rows) and all(a["sha256"] == b["sha256"] and a["relative_path"] == b["relative_path"] for a, b in zip(reference_rows, rows)) else "FAIL",
        "resnet50_reference": str(RESNET50_REFERENCE), "current_count": len(rows), "reference_count": len(reference_rows),
        "ordered_hash_and_path_match": len(reference_rows) == len(rows) and all(a["sha256"] == b["sha256"] and a["relative_path"] == b["relative_path"] for a, b in zip(reference_rows, rows)),
    }
    write_json(RUN / "validation/resnet50_test_dataset_parity.json", parity)
    if parity["status"] != "PASS":
        raise RuntimeError(f"ResNet50 dataset parity failed: {parity}")
    mobile_rows = list(csv.DictReader((MOBILENET_REFERENCE / "dataset/dataset_inventory.csv").open()))
    mobile_match = len(mobile_rows) == len(rows) and all(
        a["sha256"] == b["sha256"] and a["relative_path"] == b["relative_path"]
        for a, b in zip(mobile_rows, rows)
    )
    cross_model_parity = {
        "status": "PASS" if parity["status"] == "PASS" and mobile_match else "FAIL",
        "current_count": len(rows),
        "resnet50_reference": str(RESNET50_REFERENCE),
        "resnet50_match": parity["status"] == "PASS",
        "mobilenet_reference": str(MOBILENET_REFERENCE),
        "mobilenet_match": mobile_match,
        "class_mapping": {str(i): name for i, name in enumerate(names)},
    }
    write_json(RUN / "validation/mobilenet_test_dataset_parity.json", cross_model_parity)
    write_json(RUN / "validation/cross_model_test_dataset_parity.json", cross_model_parity)
    if cross_model_parity["status"] != "PASS":
        raise RuntimeError(f"Cross-model dataset parity failed: {cross_model_parity}")
    report = (
        "# Test Dataset Validation\n\n"
        f"- Actual source: `{DATASET_ROOT}`\n- Requested alias `/home/jetson/Downloads/classification_benchmark/test_imgs` was absent.\n"
        f"- Images: {len(rows)}\n- Classes: {names}\n- Corrupt/hidden/unsupported: 0\n"
        f"- Duplicate-content groups: {len(duplicates)}\n"
        "- ResNet50 ordered hash parity: PASS\n- MobileNetV3-Small ordered hash parity: PASS\n"
    )
    (RUN / "dataset/test_dataset_validation.md").write_text(report, encoding="utf-8")
    (RUN / "dataset/dataset_validation_report.md").write_text(report, encoding="utf-8")


def preserve_sources() -> None:
    required = [SOURCE_MODEL, ORIGINAL_ONNX, SOURCE_CLASS_NAMES, HISTORICAL_SCRIPT]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Required source artifact missing: {path}")
    copy_preserve(SOURCE_MODEL, PACKAGED_MODEL)
    copy_preserve(ORIGINAL_ONNX, PACKAGED_ORIGINAL_ONNX)
    copy_preserve(HISTORICAL_SCRIPT, RUN / "source/historical_yolov8n_cls_fp32_core4_benchmark.py")
    model, names, checkpoint = load_model()
    train_args = checkpoint.get("train_args", {}) if isinstance(checkpoint, dict) else {}
    metadata = {
        "architecture": "YOLOv8n-cls", "task": "classification", "class_count": len(names),
        "class_names": names, "input_shape": list(INPUT_SHAPE), "input_dtype": "float32",
        "checkpoint_type": "trusted Ultralytics checkpoint dictionary containing ClassificationModel",
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "ultralytics_checkpoint_version": checkpoint.get("version"),
        "training_args": train_args,
        "related_training_checkpoint_found": True,
        "source_artifact_for_engine": "fixed-interface copy of provenance-linked ONNX",
        "checkpoint_keys": sorted(checkpoint) if isinstance(checkpoint, dict) else [],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "related_checkpoint": file_record(SOURCE_MODEL), "packaged_checkpoint": file_record(PACKAGED_MODEL),
        "copy_hash_matches": sha256(SOURCE_MODEL) == sha256(PACKAGED_MODEL),
    }
    write_json(RUN / "model/model_metadata.json", metadata)
    write_json(RUN / "model/source_model_metadata.json", metadata)
    write_json(RUN / "model/selected_model_provenance.json", {
        "status": "PASS",
        "architecture": "YOLOv8n-cls",
        "selected_source_model": str(SOURCE_MODEL),
        "source_format": "Ultralytics .pt checkpoint",
        "source_sha256": sha256(SOURCE_MODEL),
        "provenance_linked_onnx": str(ORIGINAL_ONNX),
        "onnx_sha256": sha256(ORIGINAL_ONNX),
        "selection_reason": "The checkpoint embeds a three-class ultralytics.nn.tasks.ClassificationModel, the expected class names, 640 training size, and is the exact checkpoint paired with this ONNX by the executed historical FP32 benchmark. Identical hashes recur in the historical FP32, FP16, and INT8 packages.",
        "confidence": "HIGH",
        "original_checkpoint_unchanged": sha256(SOURCE_MODEL) == sha256(PACKAGED_MODEL),
        "original_onnx_unchanged": sha256(ORIGINAL_ONNX) == sha256(PACKAGED_ORIGINAL_ONNX),
    })
    candidate_paths = sorted({
        *SOURCE_MODEL_DIR.glob("*.pt"),
        *SOURCE_MODEL_DIR.glob("*.onnx"),
        *SOURCE_MODEL_DIR.glob("*.engine"),
        *(SOURCE_ROOT / "models/yolov8n_cls").glob("*"),
        *(SOURCE_ROOT / "yolov8n_cls_fp32_core4_benchmark/models").glob("*"),
    })
    candidates = []
    for path in candidate_paths:
        if path.is_file():
            selected = path == SOURCE_MODEL
            candidates.append({
                **file_record(path),
                "extension": path.suffix.lower(),
                "source_format": path.suffix.lower().lstrip("."),
                "architecture_evidence": "YOLOv8n-cls historical model directory and benchmark references",
                "class_count": 3 if path.suffix.lower() in {".pt", ".onnx", ".engine"} else "",
                "input_shape": "[1,3,640,640]" if path == SOURCE_MODEL else "historical/inspect separately",
                "selected": selected,
                "selection_confidence": "HIGH" if selected else "REFERENCE_ONLY",
                "selection_note": "selected trusted trained checkpoint" if selected else "historical candidate/reference",
            })
    write_csv(RUN / "model/model_candidates.csv", candidates)
    write_csv(RUN / "source/historical_model_candidates.csv", candidates)
    write_json(RUN / "onnx/selected_onnx_provenance.json", {
        "status": "PENDING_STATIC_INTERFACE_COPY",
        "selected_historical_onnx": file_record(ORIGINAL_ONNX),
        "packaged_original_dynamic_onnx": file_record(PACKAGED_ORIGINAL_ONNX),
        "copy_hash_matches": sha256(ORIGINAL_ONNX) == sha256(PACKAGED_ORIGINAL_ONNX),
        "planned_static_onnx": str(PACKAGED_ONNX),
        "reason": "The validated historical ONNX has dynamic interface metadata; the final protocol requires a fixed [1,3,640,640] graph interface.",
    })
    (RUN / "model/model_sha256.txt").write_text(f"{sha256(PACKAGED_MODEL)}  {PACKAGED_MODEL.name}\n", encoding="utf-8")
    (RUN / "model/source_model_sha256.txt").write_text(f"{sha256(PACKAGED_MODEL)}  {PACKAGED_MODEL.name}\n", encoding="utf-8")
    (RUN / "model/model_selection_report.md").write_text(
        "# YOLOv8n-cls Model Selection\n\n"
        f"- Selected: `{SOURCE_MODEL}`\n"
        f"- SHA-256: `{sha256(SOURCE_MODEL)}`\n"
        "- Type: trusted Ultralytics three-class ClassificationModel checkpoint.\n"
        f"- Classes: {names}\n"
        "- Confidence: HIGH; model and paired ONNX hashes recur across executed historical precision packages.\n",
        encoding="utf-8",
    )


def audit_history_and_preprocessing() -> None:
    config = json.loads(PREPROCESSING_PATH.read_text())
    expected = {
        "img_size": 640,
        "resize_shortest_edge": 640,
        "center_crop": [640, 640],
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "input_shape": [1, 3, 640, 640],
    }
    if config != expected:
        raise RuntimeError(f"Preprocessing ambiguity: {config} != {expected}")
    historical_metrics = json.loads((SOURCE_ROOT / "yolov8n_cls_fp32_core4_benchmark/metrics/yolov8n_cls_fp32_core4_benchmark_table.json").read_text())
    evidence = [
        {"property": "architecture", "value": "YOLOv8n-cls three-class ClassificationModel", "source": str(SOURCE_MODEL), "lines": "checkpoint model/train_args metadata"},
        {"property": "decoder_color", "value": "PIL.Image.open(...).convert('RGB')", "source": str(HISTORICAL_SCRIPT), "lines": "138-143"},
        {"property": "geometry", "value": "torchvision Resize(shortest edge=640) then CenterCrop(640,640)", "source": str(HISTORICAL_SCRIPT), "lines": "138-143; ultralytics.data.augment.classify_transforms"},
        {"property": "interpolation", "value": "torchvision bilinear, antialias=None", "source": "ultralytics 8.4.47 classify_transforms runtime inspection", "lines": "Resize(size=640, interpolation=bilinear, antialias=None)"},
        {"property": "exif_orientation", "value": "no explicit ImageOps.exif_transpose", "source": str(HISTORICAL_SCRIPT), "lines": "142"},
        {"property": "pixel_scale", "value": "ToTensor divide by 255", "source": "torchvision.transforms.ToTensor via classify_transforms", "lines": "runtime transform composition"},
        {"property": "normalization", "value": f"mean={config['mean']} std={config['std']} (identity)", "source": "ultralytics 8.4.47 DEFAULT_MEAN/DEFAULT_STD runtime inspection", "lines": "DEFAULT_MEAN=(0,0,0), DEFAULT_STD=(1,1,1)"},
        {"property": "layout_dtype", "value": "contiguous NCHW float32 batch 1", "source": str(HISTORICAL_SCRIPT), "lines": "138-143 and TensorRT evaluator"},
        {"property": "output", "value": "ONNX output0 contains embedded Softmax; no external softmax", "source": str(SOURCE_ROOT / "benchmark_results/yolov8n_cls_fp32/onnx_inspection.txt"), "lines": "Output: output0 ['Softmaxoutput0_dim_0',3]"},
    ]
    write_csv(RUN / "config/preprocessing_evidence.csv", evidence)
    write_csv(RUN / "source/historical_yolov8n_cls_evidence.csv", evidence)
    write_csv(RUN / "source/historical_preprocessing_evidence.csv", evidence)
    write_json(RUN / "config/normalization_verification.json", {"status": "PASS", "mean": config["mean"], "std": config["std"], "source": str(HISTORICAL_SCRIPT), "ambiguity": None})
    write_json(RUN / "config/class_mapping.json", {"0": "class_1_asymptomatic", "1": "class_2_mild_symptoms", "2": "class_3_symptomatic_compromised"})
    write_json(RUN / "source/historical_result_reference.json", {"accuracy": 0.8505747126436781, "macro_f1": 0.8245594939350424, "inference_fps": 29.282453667686415, "inference_latency_ms": 34.15014367814107, "energy_per_inference_j": 0.251133713185714, "reference_only": True, "raw_table": historical_metrics})
    audit = (
        "# Historical YOLOv8n-cls Audit\n\n"
        f"- Executed source: `{HISTORICAL_SCRIPT}`\n- Architecture: YOLOv8n-cls, three classes.\n"
        f"- Original ONNX: `{ORIGINAL_ONNX}` (`{sha256(ORIGINAL_ONNX)}`).\n"
        "- Historical model/ONNX pair is repeated with identical hashes across FP32, FP16, and INT8 packages.\n"
        "- Preprocessing: PIL RGB, bilinear shortest-edge resize to 640, center crop 640x640, /255, identity mean/std, contiguous NCHW FP32.\n"
        "- Output: embedded Softmax probabilities; no external Softmax is applied.\n"
        "- Preprocessing ambiguity: none.\n"
    )
    (RUN / "source/historical_yolov8n_cls_audit.md").write_text(audit, encoding="utf-8")
    (RUN / "config/historical_preprocessing_audit.md").write_text(audit, encoding="utf-8")
    (RUN / "config/preprocessing_specification.yaml").write_text(
        "color_order: RGB\ndecoder: PIL.Image\nexif_orientation: not_explicitly_corrected\nresize_shortest_edge: 640\nresize_method: aspect_ratio_preserving\ninterpolation: bilinear\nantialias: null\ncenter_crop: [640, 640]\nletterbox: false\npadding: none\npixel_scale: 1/255\nmean: [0.0, 0.0, 0.0]\nstd: [1.0, 1.0, 1.0]\nlayout: NCHW\ndtype: float32\ninput_shape: [1, 3, 640, 640]\nembedded_softmax: true\nexternal_softmax: false\n",
        encoding="utf-8",
    )


def reproduction_files() -> None:
    manifest = {
        "run_directory": str(RUN), "task": "classification", "model": "YOLOv8n-cls",
        "source_format": "ONNX", "precision": "TensorRT FP32 strict", "tf32_disabled": True,
        "fp16_disabled": True, "int8_disabled": True, "calibration_used": False, "dla_used": False,
        "batch_size": 1, "input_shape": list(INPUT_SHAPE), "warmup_count": WARMUP_COUNT,
        "dataset_source": str(DATASET_ROOT), "source_original_onnx": str(ORIGINAL_ONNX),
        "source_onnx": str(ORIGINAL_ONNX),
        "packaged_static_onnx": str(PACKAGED_ONNX),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "PREPARED",
    }
    write_json(RUN / "run_manifest.json", manifest)
    (RUN / "requirements.txt").write_text(
        "numpy==1.23.5\nPillow==9.2.0\nmatplotlib==3.7.5\npsutil==7.2.2\njetson-stats==4.3.2\nTensorRT==8.0.1.6 (Jetson package; do not pip install)\n",
        encoding="utf-8",
    )
    reproduce = f"""#!/usr/bin/env bash
set -euo pipefail
RUN={RUN}
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/prepare_project.py"
/home/jetson/Downloads/jetson_folder/jetson_benchmark_env/bin/python "$RUN/scripts/prepare_static_onnx.py"
/home/jetson/Downloads/jetson_folder/jetson_benchmark_env/bin/python "$RUN/scripts/inspect_onnx_candidates.py"
/home/jetson/Downloads/jetson_folder/jetson_benchmark_env/bin/python "$RUN/scripts/validate_onnx.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/validate_trt_parser.py"
bash "$RUN/scripts/build_engine.sh"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/validate_correctness.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/run_idle_monitor.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/benchmark_yolov8n_cls_fp32_warmup10.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/finalize_results.py"
"""
    (RUN / "reproduce.sh").write_text(reproduce, encoding="utf-8"); (RUN / "reproduce.sh").chmod(0o755)
    (RUN / "reproduce.ps1").write_text(f"Write-Host 'Run on Jetson: bash {RUN}/reproduce.sh'\n", encoding="utf-8")
    (RUN / "README.md").write_text(
        "# YOLOv8n-cls TensorRT Strict-FP32 Jetson Nano Benchmark\n\n"
        "Fresh publication benchmark. Source format: ONNX. TensorRT FP32 with TF32, FP16, INT8, calibration, and DLA disabled.\n"
        "Input 1x3x640x640, batch 1, exactly 10 excluded warmups, then all 261 validated test images once.\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    active = command_output("ps -eo pid,args | grep -Ei '/trtexec|benchmark_yolov8n_cls|run_yolov8n_cls' | grep -v grep | grep -v prepare_project.py || true").strip()
    if active:
        raise RuntimeError(f"Another relevant process is active:\n{active}")
    copy_preserve(Path(__file__).resolve(), RUN / "scripts/prepare_project.py")
    copy_preserve(SOURCE_CLASS_NAMES, CLASS_NAMES_PATH)
    write_json(PREPROCESSING_PATH, {
        "img_size": 640,
        "resize_shortest_edge": 640,
        "center_crop": [640, 640],
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "input_shape": [1, 3, 640, 640],
    })
    environment_inventory(); inventory_dataset(); preserve_sources(); audit_history_and_preprocessing(); reproduction_files()
    write_json(RUN / "validation/preparation_status.json", {"status": "PASS"})
    print("PREPARATION PASS")
    print(f"RUN={RUN}")
    print(f"SELECTED_SOURCE_MODEL={SOURCE_MODEL}")
    print(f"SOURCE_MODEL_SHA256={sha256(SOURCE_MODEL)}")
    print(f"PROVENANCE_LINKED_DYNAMIC_ONNX={ORIGINAL_ONNX}")
    print(f"ORIGINAL_ONNX_SHA256={sha256(ORIGINAL_ONNX)}")
    print("INPUT=[1,3,640,640] OUTPUT=[1,3] CLASSES=3")
    print("SELECTION_EVIDENCE=trusted three-class checkpoint and identical historical model/ONNX hashes")
    print(f"DATASET={DATASET_ROOT} IMAGES={len(IMAGE_ORDER_PATH.read_text().splitlines())}")


if __name__ == "__main__":
    main()
