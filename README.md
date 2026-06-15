# vec-lite

A vector database from scratch in Go. Store high-dimensional embeddings and find the
nearest ones to a query, which is the engine behind semantic search, recommendations,
and retrieval for AI systems.

This is a learning-grade system, not a production datastore. The design and the
tradeoffs behind each piece are in [DESIGN.md](DESIGN.md).

## What it does

- Three distance metrics: Euclidean, cosine, and inner product.
- Exact brute-force index, and an HNSW approximate index for sub-linear search.
- Tunable recall vs latency at query time (efSearch). Recall ~0.99 against the exact
  answer on structured data at default settings.
- Snapshot persistence, so an index survives a restart without rebuilding the graph.
- Metadata payloads with filtered search ("nearest neighbors where kind=database").
- An HTTP/JSON service and a runnable text-search demo.

Out of scope on purpose: producing embeddings (an upstream model's job), GPU/SIMD
distance kernels, and multi-node deployment (the sharding path is sketched in
DESIGN.md). See the non-goals there.

## Run the demo

Requires Go 1.26 or newer. The demo embeds a small corpus of programming terms and
queries it with misspellings:

```sh
go run ./cmd/demo
```

```
query "pyton"      -> python (0.198), pytorch (0.430), numpy (0.857)
query "kubernates" -> kubernetes (0.214), kafka (0.798), postgres (0.831)
query "javascrpt"  -> javascript (0.168), java (0.547), typescript (0.600)
nearest databases to "postgers" -> postgres, redis, mongodb
```

The trigram embedder here is lexical similarity, not learned semantics. Swap it for a
real embedding model and the same index and queries become semantic search; the
database only ever sees vectors.

## Run the HTTP service

```sh
go run ./cmd/server -addr :8081 -dim 4 -metric cosine

curl -X PUT localhost:8081/vectors/a -d '{"vector":[1,0,0,0],"metadata":{"kind":"x"}}'
curl -X PUT localhost:8081/vectors/c -d '{"vector":[0.9,0.1,0,0],"metadata":{"kind":"x"}}'
curl -X POST localhost:8081/search  -d '{"vector":[1,0,0,0],"k":3}'
curl -X POST localhost:8081/search  -d '{"vector":[1,0,0,0],"k":3,"filter":{"kind":"x"}}'
```

## Use it as a library

```go
idx := index.NewHNSW(3, vector.Cosine, index.DefaultHNSWConfig())
idx.Add("doc-1", vector.Vector{0.1, 0.2, 0.9})
hits, _ := idx.Search(vector.Vector{0.1, 0.2, 0.8}, 5)
idx.SaveFile("docs.idx") // reload later with index.LoadHNSWFile
```

## Test and benchmark

```sh
go test ./... -race
go test ./internal/index -run '^$' -bench .          # engine microbenchmarks
go run ./cmd/bench -n 20000 -dim 128 -queries 1000   # recall/latency sweep
```

Tests cover the metrics, the brute-force oracle against a full-sort reference, HNSW
recall against that oracle, the efSearch recall curve, persistence round-trips,
filtered-search recall, and the text embedder.

## Layout

```
internal/vector       float32 vectors and distance metrics
internal/index        BruteForce (the oracle), HNSW, and snapshot persistence
internal/collection   metadata payloads and filtered search
internal/embed        trigram text embedder (for the demo)
cmd/server            HTTP/JSON service
cmd/demo              end-to-end text-search demo
cmd/bench             recall/latency benchmark
DESIGN.md             design, tradeoffs, benchmarks, and the sharding sketch
```
