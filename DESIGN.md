# vec-lite: design

A vector database from scratch: store high-dimensional embeddings and answer
nearest-neighbor queries ("which stored vectors are most similar to this one").
This is what powers semantic search, recommendations, and retrieval for AI systems.

The goal is to demonstrate the core decisions of a vector database in working code:
distance metrics, an approximate nearest-neighbor index (HNSW), persistence,
metadata filtering, and a measured recall/latency tradeoff. This document grows
alongside the code; sections marked TODO are on the roadmap below.

## Status

Day 4 of 7. Done: float32 vectors and the three distance metrics, an exact
brute-force index (the oracle), the HNSW approximate index, and snapshot persistence.

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

## Approximate index, HNSW (Day 2-3, done)

Hierarchical Navigable Small World graph (`internal/index/hnsw.go`). Nodes live on
multiple layers. Each node is assigned a top layer drawn from an exponential
distribution, so layer 0 holds every node and each higher layer holds about 1/M of
the layer below. The upper layers are a sparse express lane; the bottom layer is the
dense graph.

Search starts at a single entry point on the top layer and greedily hops to the
neighbor closest to the query, descending one layer at a time with a width-1 search
until layer 0. At layer 0 it widens to an ef-sized beam search and returns the
nearest k. Insertion does the same descent, then on each layer from the node's top
down to 0 it links the new node to its closest neighbors and prunes every touched
neighbor list back to the degree cap.

Parameters and their tradeoffs:
- M (neighbors per node, default 16; layer 0 uses 2M): higher M raises recall and
  memory and slows inserts. It is the graph's degree.
- efConstruction (default 200): beam width while building. Higher means a better
  graph and slower inserts, but no query cost.
- efSearch (default 50, tunable at query time via SetEfSearch): beam width while
  querying. This is the live recall/latency dial. Tests show recall climbing from
  about 0.76 at ef=10 to 0.99 at ef=80 on the same graph, and recall@10 of about
  0.99 at the defaults against the brute-force oracle.

Neighbor selection here keeps the closest candidates. The original paper's heuristic
also keeps some farther but more diverse neighbors to improve graph connectivity and
recall on clustered data; that refinement is noted as future work. Updating an
existing id currently replaces its vector but keeps its links; a full relink on
update is also future work.

## Persistence (Day 4, done)

`Save`/`Load` snapshot the whole index (`internal/index/persist.go`). The snapshot
stores the finished graph (each node's vector and its per-layer neighbor lists), not
just the raw vectors, because building the graph is the expensive part: every insert
runs a search, so rebuilding a large index from vectors alone would be slow. Reloading
restores the graph directly and skips that cost. A round-trip test confirms a loaded
index returns identical search results and still accepts new inserts.

Format is encoding/gob over parallel arrays of exported fields, which keeps the codec
trivial. `SaveFile` writes to a temp file and renames, so a crash mid-save cannot
replace a good snapshot with a half-written one.

Chosen tradeoff: a full snapshot is simple and the load is a single read, but it
rewrites everything each time. An append-only log of inserts would make saves
incremental at the cost of replay-on-load and compaction. For an index that is
rebuilt or snapshotted periodically rather than on every write, the snapshot is the
simpler fit; the log approach is noted as the alternative.

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
- [x] Day 2-3: HNSW approximate index
- [x] Day 4: persistence (save/load an index)
- [ ] Day 5: metadata payloads and filtered search
- [ ] Day 6: recall/latency benchmarks and parameter tuning
- [ ] Day 7: HTTP service, a semantic-search demo, and a sharding sketch
