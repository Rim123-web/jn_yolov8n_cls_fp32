from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import onnx


RUN = Path(__file__).resolve().parents[1]
SOURCE = RUN / "onnx/yolov8n_cls_original_dynamic.onnx"
OUTPUT = RUN / "onnx/yolov8n_cls_source.onnx"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def set_shape(value, dimensions: list[int]) -> None:
    shape = value.type.tensor_type.shape
    if len(shape.dim) != len(dimensions):
        raise RuntimeError(f"Rank mismatch for {value.name}: {len(shape.dim)}")
    for dim, size in zip(shape.dim, dimensions):
        dim.ClearField("dim_param")
        dim.dim_value = size


start = time.monotonic()
model = onnx.load(str(SOURCE))
onnx.checker.check_model(model)
if len(model.graph.input) != 1 or model.graph.input[0].name != "images":
    raise RuntimeError(f"Unexpected original ONNX inputs: {[x.name for x in model.graph.input]}")
if len(model.graph.output) != 1 or model.graph.output[0].name != "output0":
    raise RuntimeError(f"Unexpected original ONNX outputs: {[x.name for x in model.graph.output]}")
set_shape(model.graph.input[0], [1, 3, 640, 640])
set_shape(model.graph.output[0], [1, 3])
onnx.save(model, str(OUTPUT))
fixed = onnx.load(str(OUTPUT))
onnx.checker.check_model(fixed)
duration = time.monotonic() - start
operators = Counter(node.op_type for node in fixed.graph.node)
metadata = {
    "status": "PASS",
    "operation": "static-interface copy of the validated historical ONNX; graph nodes, initializers, and weights were not re-exported",
    "original_path": str(SOURCE),
    "original_size_bytes": SOURCE.stat().st_size,
    "original_sha256": digest(SOURCE),
    "output_path": str(OUTPUT),
    "output_size_bytes": OUTPUT.stat().st_size,
    "output_sha256": digest(OUTPUT),
    "duration_seconds": duration,
    "onnx_version": onnx.__version__,
    "opset": [item.version for item in fixed.opset_import],
    "input": {"name": "images", "shape": [1, 3, 640, 640], "dtype": "FLOAT"},
    "output": {"name": "output0", "shape": [1, 3], "dtype": "FLOAT"},
    "dynamic_shapes": False,
    "embedded_preprocessing": False,
    "embedded_softmax": operators.get("Softmax", 0) > 0,
    "node_count": len(fixed.graph.node),
    "initializer_count": len(fixed.graph.initializer),
    "operator_counts": dict(sorted(operators.items())),
    "onnx_checker": "PASS",
    "onnxruntime_validation": "UNAVAILABLE_ON_JETSON_UNSAFE_RUNTIME_EXCLUDED",
}
(RUN / "onnx/onnx_export_config.json").write_text(
    json.dumps({
        "method": "staticize_existing_validated_onnx_interface",
        "source_checkpoint_unchanged": True,
        "source_onnx_unchanged": True,
        "input_shape": [1, 3, 640, 640],
        "output_shape": [1, 3],
        "fp16": False,
        "int8": False,
        "dynamic": False,
    }, indent=2) + "\n",
    encoding="utf-8",
)
(RUN / "onnx/onnx_export_duration.json").write_text(
    json.dumps({"duration_seconds": duration, "note": "Interface staticization only; no PyTorch re-export."}, indent=2) + "\n",
    encoding="utf-8",
)
(RUN / "onnx/onnx_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
(RUN / "onnx/onnx_graph_summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
(RUN / "onnx/onnx_sha256.txt").write_text(f"{metadata['output_sha256']}  {OUTPUT.name}\n", encoding="utf-8")
(RUN / "onnx/onnx_validation_report.md").write_text(
    "# ONNX Validation\n\n"
    "- Original historical ONNX preserved unchanged: YES\n"
    "- Static-interface copy created without changing graph nodes or weights: YES\n"
    "- onnx.checker: PASS\n"
    "- Input: `images`, FLOAT, `[1,3,640,640]`\n"
    "- Output: `output0`, FLOAT, `[1,3]`\n"
    f"- Embedded Softmax: {metadata['embedded_softmax']}\n"
    "- ONNX Runtime: unavailable/unsafe on this Jetson; excluded rather than forced.\n",
    encoding="utf-8",
)
print(json.dumps(metadata, indent=2))
print("STATIC ONNX PREPARATION PASS")
