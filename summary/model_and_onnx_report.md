# Historical YOLOv8n-cls Audit

- Executed source: `/home/jetson/Downloads/classification_jetson/yolov8n_cls_fp32_core4_benchmark/scripts/run_yolov8n_cls_fp32_core4_benchmark.py`
- Architecture: YOLOv8n-cls, three classes.
- Original ONNX: `/home/jetson/Downloads/classification_jetson/yolov8n_cls_inference_models/yolov8n_cls_inference.onnx` (`8c4fa4a682da136cf09bb0bbc18f88d05d94b4f6d779771c892152824ae24c64`).
- Historical model/ONNX pair is repeated with identical hashes across FP32, FP16, and INT8 packages.
- Preprocessing: PIL RGB, bilinear shortest-edge resize to 640, center crop 640x640, /255, identity mean/std, contiguous NCHW FP32.
- Output: embedded Softmax probabilities; no external Softmax is applied.
- Preprocessing ambiguity: none.

# ONNX Validation

- onnx.checker: PASS
- Fixed FP32 input: [1,3,640,640]
- FP32 output: [1,3]
- ONNX Runtime CPU dummy inference: unavailable on this Jetson and not forced
- Embedded Softmax: True
