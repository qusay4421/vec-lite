# vec-lite

A vector database from scratch in Go. Store high-dimensional embeddings and find the
nearest ones to a query, which is the engine behind semantic search, recommendations,
and retrieval for AI systems.

This is a learning-grade system, not a production datastore. The design and the
tradeoffs behind each piece are in [DESIGN.md](DESIGN.md).

## What works today

Day 1 of a one-week build. So far:
- float32 vectors with three distance metrics: Euclidean, cosine, and inner product.
- An exact brute-force index that returns the k nearest vectors, using a bounded
  max-heap so a query is O(N log k).

The brute-force index is also the correctness oracle for the approximate (HNSW) index
that comes next: an approximate search is judged by how often it agrees with the exact
answer. The roadmap is in DESIGN.md.

## Use it as a library

```go
import (
    "github.com/qusay4421/vec-lite/internal/index"
    "github.com/qusay4421/vec-lite/internal/vector"
)

idx := index.NewBruteForce(3, vector.Cosine)
idx.Add("doc-1", vector.Vector{0.1, 0.2, 0.9})
idx.Add("doc-2", vector.Vector{0.8, 0.1, 0.1})

hits, _ := idx.Search(vector.Vector{0.1, 0.2, 0.8}, 1)
// hits[0].ID is the closest stored vector; hits[0].Score is its distance.
```

## Test

```sh
go test ./... -race
```

Covers each distance metric (including the orthogonal, opposite, and zero-vector edge
cases), and the brute-force index against an independent full-sort reference on random
data.

## Layout

```
internal/vector    float32 vectors and distance metrics
internal/index     search indexes; BruteForce exact index (the oracle)
DESIGN.md          design, the ANN problem, tradeoffs, and roadmap
```

## Where it's going

HNSW approximate search, persistence, metadata filtering, measured recall-vs-latency
numbers, and an HTTP service with a semantic-search demo. See the roadmap in DESIGN.md.
