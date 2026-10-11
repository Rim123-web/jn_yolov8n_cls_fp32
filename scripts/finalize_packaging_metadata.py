from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN = Path(__file__).resolve().parents[1]
TIMING = RUN / "logs/archive_timing_probe.txt"


match = re.search(r"elapsed_seconds=([0-9.]+)", TIMING.read_text(encoding="utf-8"))
if not match:
    raise SystemExit("Packaging timing probe value is missing")
packaging_seconds = float(match.group(1))
path = RUN / "summary/workflow_phase_durations.json"
workflow = json.loads(path.read_text(encoding="utf-8"))
for row in workflow["phases"]:
    if row["phase"] == "Packaging":
        row["duration_seconds"] = packaging_seconds
        row["notes"] = "Measured full-package tar.gz timing probe before immutable final archive creation."
    elif row["phase"] == "Transfer and local verification":
        row["duration_seconds"] = "RECORDED_IN_EXTERNAL_TRANSFER_VERIFICATION"
        row["notes"] = "Exact final transfer duration is necessarily recorded beside the archive after transfer; no value was guessed inside the pre-transfer archive."
workflow["instrumented_total_seconds"] = sum(
    float(row["duration_seconds"])
    for row in workflow["phases"]
    if isinstance(row["duration_seconds"], (int, float))
)
path.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")

with (RUN / "summary/workflow_phase_durations.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["phase", "duration_seconds", "notes"])
    writer.writeheader()
    writer.writerows(workflow["phases"])
(RUN / "summary/workflow_duration_report.md").write_text(
    "# Workflow Duration\n\n"
    + "\n".join(
        f"- {row['phase']}: {row['duration_seconds']} s ({row['notes']})"
        for row in workflow["phases"]
    )
    + "\n",
    encoding="utf-8",
)

with (RUN / "plot_data/workflow_phase_durations.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["phase", "duration_seconds", "notes"])
    writer.writeheader()
    writer.writerows(workflow["phases"])
numeric = [row for row in workflow["phases"] if isinstance(row["duration_seconds"], (int, float))]
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh([row["phase"] for row in numeric][::-1], [row["duration_seconds"] for row in numeric][::-1])
ax.set(xlabel="Duration (seconds)", title="Instrumented Workflow Phase Durations")
for extension in ("png", "svg", "pdf"):
    kwargs = {"dpi": 300} if extension == "png" else {}
    fig.savefig(RUN / f"figures/{extension}/workflow_phase_durations.{extension}", bbox_inches="tight", **kwargs)
plt.close(fig)
print(f"PACKAGING METADATA PASS: {packaging_seconds:.6f} seconds")
