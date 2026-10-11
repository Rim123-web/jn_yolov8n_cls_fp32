from __future__ import annotations

import json
from pathlib import Path

import onnx


RUN = Path(__file__).resolve().parents[1]
ONNX_PATH = RUN / "onnx/yolov8n_cls_source.onnx"


def shape(value):
    return [dimension.dim_value if dimension.dim_value else dimension.dim_param for dimension in value.type.tensor_type.shape.dim]


model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(model)
inputs = [
    {"name": value.name, "shape": shape(value), "elem_type": value.type.tensor_type.elem_type}
    for value in model.graph.input
]
outputs = [
    {"name": value.name, "shape": shape(value), "elem_type": value.type.tensor_type.elem_type}
    for value in model.graph.output
]
if len(inputs) != 1 or inputs[0]["shape"] != [1, 3, 640, 640]:
    raise RuntimeError(f"Unexpected ONNX input: {inputs}")
if len(outputs) != 1 or outputs[0]["shape"] != [1, 3]:
    raise RuntimeError(f"Unexpected ONNX output: {outputs}")
metadata = {
    "status": "PASS",
    "onnx_version": onnx.__version__,
    "onnxruntime_version": None,
    "providers": [],
    "onnxruntime_status": "UNAVAILABLE_ON_JETSON_UNSAFE_RUNTIME_EXCLUDED",
    "opset": [item.version for item in model.opset_import],
    "inputs": inputs,
    "outputs": outputs,
    "node_count": len(model.graph.node),
    "dummy_output_shape": None,
    "dynamic_shapes": False,
    "embedded_preprocessing": False,
    "embedded_softmax": any(node.op_type == "Softmax" for node in model.graph.node),
}
(RUN / "onnx/onnx_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
(RUN / "onnx/onnx_graph_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
(RUN / "onnx/onnx_validation_report.md").write_text(
    "# ONNX Validation\n\n"
    "- onnx.checker: PASS\n"
    "- Fixed FP32 input: [1,3,640,640]\n"
    "- FP32 output: [1,3]\n"
    "- ONNX Runtime CPU dummy inference: unavailable on this Jetson and not forced\n"
    f"- Embedded Softmax: {metadata['embedded_softmax']}\n"
)
import hashlib
digest = hashlib.sha256(ONNX_PATH.read_bytes()).hexdigest()
(RUN / "onnx/onnx_sha256.txt").write_text(f"{digest}  {ONNX_PATH.name}\n")
print("ONNX VALIDATION PASS")
