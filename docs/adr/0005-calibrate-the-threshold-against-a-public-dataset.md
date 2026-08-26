# 5. Calibrate the match threshold against a public dataset

Date: 2026-08-26

## Status

Accepted

## Context

The match threshold is the single number deciding whether the system greets someone
by name or reports them as an intruder. No value is correct in the abstract: it
depends on the embedding model, the cameras, and who is enrolled.

The obvious approach is to calibrate on the household's own photographs. That fails
for a measurement reason rather than a philosophical one. The figure that matters is
the false-accept rate — how often a stranger is wrongly named — and with two or three
enrolled people there are only a handful of different-person pairs to measure it from.
A rate of one in a thousand cannot be observed in a sample of thirty.

## Decision

Measure against Labeled Faces in the Wild: roughly 13 000 photographs of 5 749 people,
downloaded on demand into the git-ignored data directory. `reconvision eval` reports
the score distributions for same-person and different-person pairs, and the threshold
that admits at most a chosen share of the latter. Enrolled identities are included in
the measurement when present.

## Consequences

- The reported false-accept rate is meaningful, and comparable to published figures for
  the same embedding model.
- LFW is not a home camera. It is well-lit, roughly frontal, and skews heavily towards
  public figures of a particular demographic. The threshold it yields is a starting
  point calibrated on easier images than a hallway at night will produce, so real
  performance will be worse than the number suggests. The correction loop exists to
  close that gap: confirmed events add real captures from the real cameras, and a later
  `eval` measures against those too.
- Calibration needs a 243 MB download and several minutes of encoding on first run. It
  is a setup step, not something on the recognition path.
- Operating points are computed by counting sorted impostor scores rather than by
  interpolating a quantile, which returned a value between two samples and let slightly
  more impostors through than the rate promised. The printed number is a commitment to
  the user and has to hold exactly.
