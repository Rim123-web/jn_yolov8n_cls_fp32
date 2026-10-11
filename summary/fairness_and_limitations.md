# Fairness and Limitations

- This is a single-device, batch-1 deployment benchmark.
- TensorRT was built locally for this Jetson and is not portable across TensorRT/GPU versions.
- Power TOT is module-level input power, not GPU-only power.
- jtop sampling is slower than per-image inference stages; per-image power pairing is nearest-sample only.
- CUDA event synchronization improves stage attribution but can reduce throughput relative to an asynchronous pipeline.
- YOLOv8n-cls shortest-edge resize and center crop preserve aspect ratio before cropping; this differs from the recovered native pipelines of the comparison architectures.
