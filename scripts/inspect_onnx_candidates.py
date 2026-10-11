from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import onnx


RUN = Path(__file__).resolve().parents[1]
CANDIDATES = [
    Path("/home/jetson/Downloads/classification_jetson/yolov8n_cls_inference_models/yolov8n_cls_inference.onnx"),
    RUN / "onnx/yolov8n_cls_original_dynamic.onnx",
    RUN / "onnx/yolov8n_cls_source.onnx",
]
SELECTED = CANDIDATES[2]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def shape(value):
    return [d.dim_value if d.dim_value else d.dim_param for d in value.type.tensor_type.shape.dim]


rows = []
for path in CANDIDATES:
    if not path.is_file():
        continue
    status, error = "PASS", ""
    try:
        model = onnx.load(str(path)); onnx.checker.check_model(model)
    except Exception as exc:
        status, error = "FAIL", repr(exc); model = None
    inputs = [{"name": v.name, "shape": shape(v), "dtype": v.type.tensor_type.elem_type} for v in model.graph.input] if model else []
    outputs = [{"name": v.name, "shape": shape(v), "dtype": v.type.tensor_type.elem_type} for v in model.graph.output] if model else []
    nodes = [node.op_type for node in model.graph.node] if model else []
    rows.append({
        "path": str(path), "filename": path.name, "size_bytes": path.stat().st_size,
        "modified_time": path.stat().st_mtime, "sha256": digest(path), "onnx_checker": status,
        "checker_error": error, "opset": json.dumps([item.version for item in model.opset_import]) if model else "",
        "input_name": inputs[0]["name"] if inputs else "", "input_shape": json.dumps(inputs[0]["shape"]) if inputs else "",
        "input_dtype": inputs[0]["dtype"] if inputs else "", "output_name": outputs[0]["name"] if outputs else "",
        "output_shape": json.dumps(outputs[0]["shape"]) if outputs else "", "output_dtype": outputs[0]["dtype"] if outputs else "",
        "class_count": outputs[0]["shape"][-1] if outputs and outputs[0]["shape"] else "",
        "dynamic": any(isinstance(v, str) and v for item in inputs + outputs for v in item["shape"]),
        "embedded_preprocessing": False, "embedded_softmax": "Softmax" in nodes,
        "hardswish_nodes": sum(name.lower() in {"hardswish", "hard_swish"} for name in nodes),
        "node_count": len(nodes), "selected": path == SELECTED,
        "historical_reference": "selected fixed-interface copy" if path == SELECTED else "preserved historical source",
    })

with (RUN / "onnx/onnx_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
(RUN / "onnx/onnx_candidate_metadata.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
selected = next(row for row in rows if row["selected"])
if selected["onnx_checker"] != "PASS" or json.loads(selected["input_shape"]) != [1, 3, 640, 640] or json.loads(selected["output_shape"]) != [1, 3]:
    raise RuntimeError(f"Selected ONNX validation failed: {selected}")
print(json.dumps(selected, indent=2))
print("ONNX CANDIDATE INSPECTION PASS")
