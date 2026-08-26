# 3. SQLite with brute-force vector search

Date: 2026-08-26

## Status

Accepted

## Context

The system stores two kinds of data: recognition events with their snapshots, which
grow without bound, and face embeddings, which are queried on every recognised face.
The reflex for "vector search" is a vector database or PostgreSQL with pgvector and an
HNSW index.

The realistic scale here is a household: fewer than 100 identities, 10-40 embeddings
each, so at most a few thousand 512-dimensional vectors. Events may reach millions of
rows over years.

## Decision

Use SQLite in WAL mode behind an `EventRepository` / `GalleryRepository` port, with
versioned migrations. Match embeddings by brute-force cosine similarity over a NumPy
matrix held in memory.

## Consequences

- At this scale a single NumPy matrix multiply over a few thousand vectors is faster
  than any approximate index, and it is exact. An HNSW index would add a dependency, a
  service and an approximation in exchange for a slower answer.
- No extra container on the NAS, no connection pool, no backup story beyond copying a
  file.
- SQLite's single-writer model is fine: one process writes events, and write volume is
  a handful per minute.
- If the scale assumption is ever wrong — a shared deployment, tens of thousands of
  identities — the ports confine the change to `adapters/storage/`. The domain performs
  matching on plain arrays and does not know where they came from.
