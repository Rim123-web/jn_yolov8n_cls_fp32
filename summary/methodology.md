# Methodology

- Device: NVIDIA Jetson Nano Developer Kit / Tegra X1.
- Model: YOLOv8n-cls classification from `/home/jetson/Downloads/classification_jetson/yolov8n_cls_inference_models/yolov8n_cls_inference_only.pt` (trusted `.pt` checkpoint).
- Source SHA-256: `6b756b9724c65520f10586087e119afa6cccc28a258916390d5eb641b8f83a54`.
- Source artifact: the exact historical dynamic ONNX, preserved unchanged, plus a metadata-only static-shape copy with identical graph nodes and initializers.
- Static ONNX: `/home/jetson/Downloads/classification_benchmark/yolov8n-cls/yolov8n_cls_tensorrt_fp32_640x640_batch1_warmup10_20260717_161313/onnx/yolov8n_cls_source.onnx`; SHA-256 `341afc3adf731e3ff5ac1ccc0c5893d6eed45eca47d9eb80d536ddcbde48ea1c`; opset 17; staticization 0.171577 s.
- ONNX input/output: `images` FLOAT `[1,3,640,640]` to `output0` FLOAT `[1,3]`; embedded preprocessing false; embedded softmax true.
- TensorRT strict FP32 (`--noTF32`, FP16 false, INT8 false, no calibration, no DLA).
- Fresh engine build duration: 218.559941 s; input/output binding dtypes `DataType.FLOAT` / `DataType.FLOAT`.
- Input: batch 1, RGB `[1,3,640,640]`.
- Preprocessing: PIL decode and RGB conversion, bilinear shortest-edge resize to 640,
  center crop 640x640, divide by 255, identity mean `[0,0,0]` and std `[1,1,1]`, HWC-to-CHW.
- Antialias setting: `None` in the recovered torchvision transform; no EXIF transpose, letterbox, or padding.
- Output handling: consume embedded-softmax probabilities directly, then argmax; no external softmax.
- Warm-up: exactly 10 engine executions, excluded from all official timing and monitoring.
- Measured pass: 261 validated images exactly once.
- CPU stages: `time.perf_counter_ns()`.
- H2D, TensorRT, D2H: CUDA events with synchronization at stage boundaries.
- Power boundary: Jetson module-level jtop Power TOT (POM_5V_IN), sampled at 0.5 s.
- Energy: trapezoidal integration of timestamped Power TOT samples.
- Cross-model comparison protocol parity: PASS_WITH_DOCUMENTED_NATIVE_PREPROCESSING_DIFFERENCE.
