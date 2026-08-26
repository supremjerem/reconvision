# 2. ONNX Runtime instead of PyTorch at runtime

Date: 2026-08-26

## Status

Accepted

## Context

Two models run per frame budget: an object detector (YOLO11) and a face stack (SCRFD +
ArcFace). Ultralytics ships YOLO as PyTorch weights, and the obvious path is to depend
on `ultralytics` at runtime.

PyTorch is roughly 2 GB installed. The deployment target is a container on a home NAS
whose CPU architecture is not yet known — it may be x86_64 or aarch64. Development
happens on an Apple M4 Pro, where the useful accelerator is CoreML, not CUDA.

## Decision

Export YOLO11 to ONNX once, at development time, via a `reconvision export-models`
command that lives behind the `export` optional dependency group. The runtime depends
only on `onnxruntime`. InsightFace already distributes ONNX weights.

## Consequences

- The runtime image carries no PyTorch. Image size and cold-start both drop sharply.
- One inference API covers every target: the CoreML execution provider on macOS, the CPU
  provider in the container, CUDA later if a GPU appears. `onnxruntime` publishes wheels
  for macOS arm64 and manylinux x86_64 and aarch64, so the NAS architecture is a
  non-issue.
- Model export becomes an explicit, reproducible step rather than an implicit download,
  and exported weights are checksummed.
- Changing detector architecture requires re-running the export step; we cannot swap a
  `.pt` file in place. This is an acceptable price for the deployment simplification.
