package index

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/qusay4421/vec-lite/internal/vector"
)

// buildIndex fills an HNSW with n random vectors for the search benchmarks.
func buildIndex(b *testing.B, n, dim int) (*HNSW, []vector.Vector) {
	b.Helper()
	rng := rand.New(rand.NewSource(1))
	h := NewHNSW(dim, vector.Euclidean, DefaultHNSWConfig())
	for i := 0; i < n; i++ {
		h.Add(fmt.Sprintf("v%d", i), randVec(rng, dim))
	}
	queries := make([]vector.Vector, 1000)
	for i := range queries {
		queries[i] = randVec(rng, dim)
	}
	return h, queries
}

// BenchmarkHNSWSearch measures query latency on a built graph (the path users hit).
func BenchmarkHNSWSearch(b *testing.B) {
	h, queries := buildIndex(b, 20000, 128)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Search(queries[i%len(queries)], 10)
	}
}

// BenchmarkBruteForceSearch is the linear-scan baseline the index is compared against.
func BenchmarkBruteForceSearch(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	bf := NewBruteForce(128, vector.Euclidean)
	for i := 0; i < 20000; i++ {
		bf.Add(fmt.Sprintf("v%d", i), randVec(rng, 128))
	}
	query := randVec(rng, 128)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bf.Search(query, 10)
	}
}
