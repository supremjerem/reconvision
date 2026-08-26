# 4. Modular monolith with in-process queues

Date: 2026-08-26

## Status

Accepted

## Context

Several camera streams must be decoded and analysed concurrently. A distributed design
would decouple ingestion from inference through a broker such as Redis Streams, giving
horizontal scaling and explicit back-pressure.

The deployment is a single NAS handling a handful of cameras. The models are around
400 MB of weights, and loading them once per worker process would dominate memory.

## Decision

Run a single process: one decoding thread per camera, bounded queues, and a shared pool
of inference workers. ONNX Runtime releases the GIL during inference, so threads give
real parallelism for the expensive part.

## Consequences

- Model weights are loaded once and shared across all cameras.
- Bounded queues give back-pressure for free: when inference falls behind, the decoder
  drops frames rather than accumulating latency. Dropping frames is the correct
  behaviour for live recognition, where a stale frame has no value.
- No broker to operate, and no distributed failure modes to debug.
- A crash takes down every camera at once. Mitigated by supervising the container and by
  isolating per-camera failures inside the pipeline, but it remains a real trade-off
  against a process-per-camera design.
- The port boundaries mean extracting an inference service later is a mechanical change
  if the camera count ever outgrows one machine.
