#!/usr/bin/env bash
set -euo pipefail
RUN=/home/jetson/Downloads/classification_benchmark/yolov8n-cls/yolov8n_cls_tensorrt_fp32_640x640_batch1_warmup10_20260717_161313
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
