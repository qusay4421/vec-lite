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

Requires Go 1.26 or newer. Two modes, same index and query code underneath.

Semantic search (free, offline). Download pretrained GloVe vectors once, then query by
meaning. The query words are not in the corpus; they are retrieved by meaning alone:

```sh
sh scripts/get-glove.sh                                   # ~66MB, no account needed
go run ./cmd/demo -vectors vectors/glove-wiki-gigaword-50.gz
```

```
query "monarch"    -> throne (0.235), king (0.281), queen (0.323)
query "automobile" -> car (0.304), bicycle (0.334), truck (0.345)
query "feline"     -> kitten (0.373), puppy (0.435), cat (0.445)
nearest food to "beverage" -> coffee (0.242), tea (0.342), pizza (0.390)
```

Lexical fallback (zero setup). Without a vectors file it matches by spelling, which is
enough to show typo tolerance:

```sh
go run ./cmd/demo
# query "pyton" -> python, "javascrpt" -> javascript, ...
```

The embedder is the only thing that changes between modes. The database only ever sees
vectors, so trigrams, GloVe, or a hosted model all plug into the same index.

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
internal/embed        trigram and GloVe word-vector embedders (for the demo)
cmd/server            HTTP/JSON service
cmd/demo              end-to-end text-search demo
cmd/bench             recall/latency benchmark
DESIGN.md             design, tradeoffs, benchmarks, and the sharding sketch
```
