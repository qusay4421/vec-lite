# vec-lite: design

A vector database from scratch: store high-dimensional embeddings and answer
nearest-neighbor queries ("which stored vectors are most similar to this one").
This is what powers semantic search, recommendations, and retrieval for AI systems.

The goal is to demonstrate the core decisions of a vector database in working code:
distance metrics, an approximate nearest-neighbor index (HNSW), persistence,
metadata filtering, and a measured recall/latency tradeoff. This document grows
alongside the code; sections marked TODO are on the roadmap below.

## Status

Day 1 of 7. Done: fixed-dimension float32 vectors, the three standard distance
metrics, and an exact brute-force index that doubles as the correctness oracle for
the approximate index.

## The problem

Given N vectors in D dimensions (D is often 384, 768, 1536 for real embeddings) and
a query vector, return the k most similar stored vectors. Exact search is a linear
scan: O(N*D) per query. That is fine for thousands of vectors and far too slow for
millions, which is the entire reason approximate nearest-neighbor (ANN) indexes
exist. They trade a small amount of recall (occasionally missing a true neighbor)
for orders-of-magnitude faster queries.

## Goals and non-goals

Goals:
- Exact and approximate nearest-neighbor search over float32 embeddings.
- Sub-linear query time at high recall via an HNSW graph index.
- Durable: an index survives a restart.
- Metadata payloads with filtered search ("nearest neighbors where category=shoes").
- Honest, measured recall-vs-latency numbers.

Non-goals (kept out on purpose to bound scope):
- Training or producing embeddings. vec-lite indexes vectors it is given; embedding
  models are upstream.
- GPU acceleration and SIMD-tuned distance kernels.
- A distributed, multi-node deployment. The sharding path (reusing consistent
  hashing) is sketched as future work, not built here.

## Distance metrics (Day 1, done)

Three metrics in `internal/vector`, all returned as a distance where lower means
closer so any index ranks candidates uniformly:
- Euclidean (L2): straight-line distance.
- Cosine: 1 - cosine similarity; compares direction, not magnitude, which is what
  most text embeddings want. Undefined for a zero vector, which is treated as maximally
  far rather than dividing by zero.
- Inner product: negative dot product, for maximum-inner-product search.

Vectors are float32, not float64: the index is dominated by raw vector bytes, so
halving their size matters more than the precision does. A Normalize helper is
provided because, on unit vectors, cosine distance reduces to a cheaper dot product.

## Exact index (Day 1, done)

`BruteForce` scans every vector per query and keeps the best k in a bounded max-heap,
so a query is O(N log k) time and O(k) memory rather than sorting all N distances.
It is exact by construction, which is why it serves two roles: a real index for small
datasets, and the oracle the approximate index is graded against (recall = fraction
of brute-force neighbors the ANN index also returns). A property test confirms its
top-k matches an independent full-sort reference on random data.

## Approximate index, HNSW (Day 2-3, TODO)

Hierarchical Navigable Small World graph. A layered proximity graph where search
greedily hops toward the query, starting coarse at the top layer and refining down.
To document: the M (neighbors per node) and efConstruction/efSearch parameters and
how they trade recall against speed and memory.

## Persistence (Day 4, TODO)

Save and reload an index so it survives a restart, without rebuilding the graph from
scratch. To decide: snapshot the whole graph vs an append log of inserts.

## Metadata and filtered search (Day 5, TODO)

Attach a payload to each vector and filter by it during search. To document: the
hard part, which is filtering inside graph traversal without wrecking recall.

## Benchmarks: recall and latency (Day 6, TODO)

Measure ANN recall against the brute-force oracle, plus query latency and build time,
and tune the HNSW parameters. The headline tradeoff of any vector database.

## Service and demo (Day 7, TODO)

An HTTP API and a real demo: semantic search over a text corpus using precomputed
embeddings. Plus a written sketch of how to shard across nodes with consistent
hashing, reusing the dynamo-lite ring.

## Roadmap

- [x] Day 1: vectors, distance metrics, exact brute-force index (the oracle)
- [ ] Day 2-3: HNSW approximate index
- [ ] Day 4: persistence (save/load an index)
- [ ] Day 5: metadata payloads and filtered search
- [ ] Day 6: recall/latency benchmarks and parameter tuning
- [ ] Day 7: HTTP service, a semantic-search demo, and a sharding sketch
