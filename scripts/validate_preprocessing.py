from __future__ import annotations

import json
import time

import numpy as np
from PIL import Image
from ultralytics.data.augment import classify_transforms

from common import RUN, load_image_order, preprocess, write_csv, write_json


start = time.monotonic()
samples = load_image_order()
indices = sorted({round(i * (len(samples) - 1) / 19) for i in range(20)})
historical = classify_transforms(size=640)
rows = []
for index in indices:
    path = samples[index]["path"]
    canonical = preprocess(path)
    reference = historical(Image.open(path).convert("RGB")).unsqueeze(0).contiguous().numpy()
    difference = np.abs(canonical - reference)
    rows.append({
        "sample_index": index,
        "image_path": path,
        "canonical_shape": json.dumps(list(canonical.shape)),
        "reference_shape": json.dumps(list(reference.shape)),
        "canonical_dtype": str(canonical.dtype),
        "reference_dtype": str(reference.dtype),
        "canonical_min": float(canonical.min()),
        "canonical_max": float(canonical.max()),
        "canonical_mean": float(canonical.mean()),
        "canonical_std": float(canonical.std()),
        "reference_min": float(reference.min()),
        "reference_max": float(reference.max()),
        "reference_mean": float(reference.mean()),
        "reference_std": float(reference.std()),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "finite": bool(np.isfinite(canonical).all() and np.isfinite(reference).all()),
    })
status = "PASS" if all(
    row["canonical_shape"] == "[1, 3, 640, 640]"
    and row["canonical_dtype"] == "float32"
    and row["finite"]
    and row["max_abs_difference"] <= 1e-7
    for row in rows
) else "FAIL"
summary = {
    "status": status,
    "sample_count": len(rows),
    "sampling": "20 evenly spaced deterministic images",
    "reference": "ultralytics 8.4.47 classify_transforms(size=640)",
    "maximum_absolute_difference": max(row["max_abs_difference"] for row in rows),
    "mean_absolute_difference": float(np.mean([row["mean_abs_difference"] for row in rows])),
    "duration_seconds": time.monotonic() - start,
}
write_csv(RUN / "validation/preprocessing_parity.csv", rows)
write_json(RUN / "validation/preprocessing_parity.json", summary)
(RUN / "summary/preprocessing_report.md").write_text(
    "# YOLOv8n-cls Preprocessing\n\n"
    f"- Parity status: **{status}**\n"
    "- Decoder: PIL, explicit RGB conversion, no explicit EXIF transpose.\n"
    "- Geometry: shortest-edge resize to 640, bilinear antialias=None, then center crop 640x640.\n"
    "- Values: float32 / 255, mean [0,0,0], std [1,1,1], contiguous NCHW batch 1.\n"
    f"- Maximum absolute difference across 20 images: {summary['maximum_absolute_difference']:.9g}\n",
    encoding="utf-8",
)
if status != "PASS":
    raise RuntimeError(f"Preprocessing parity failed: {summary}")
print(json.dumps(summary, indent=2))
print("PREPROCESSING PARITY PASS")
