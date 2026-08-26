# 1. Use pre-trained embeddings, not a trained classifier

Date: 2026-08-26

## Status

Accepted

## Context

The requirement is to recognise a small number of known people on home camera streams.
The instinctive reading of "recognise faces" is "train a model on photos of those
people" — a classifier with one output class per person.

Training a face recognition backbone from scratch requires millions of labelled faces
and weeks of GPU time. Fine-tuning a classifier on a handful of household members is
cheaper but has three concrete problems: it overfits badly on 10-20 photos per person,
it has no principled way to answer "this is nobody I know", and adding a person means
retraining and redeploying.

## Decision

Use a pre-trained ArcFace model (InsightFace `buffalo_l`) as a fixed feature extractor
that maps any face to a 512-dimensional unit vector. Enrolment stores the vectors of a
person's photos as a *gallery*. Recognition is cosine similarity against that gallery,
compared to a threshold calibrated on real data.

## Consequences

- Adding or removing a person is a database write, not a training run.
- "Unknown" is a first-class outcome: it is simply a best similarity below threshold.
- The system's accuracy is governed by the threshold, the face-quality filter and the
  temporal vote — not by any training we control. Those three are therefore where the
  engineering effort and the measurement go, hence `reconvision eval`.
- We inherit the pre-trained model's biases and blind spots and cannot correct them by
  training. We compensate where we can by enrolling real captures from the actual
  cameras via the correction loop, rather than relying on well-lit portraits.
- No GPU is required, so the whole system runs on a NAS.
