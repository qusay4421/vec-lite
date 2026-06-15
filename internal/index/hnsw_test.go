package index

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/qusay4421/vec-lite/internal/vector"
)

// HNSW must satisfy the same Index contract as BruteForce.
var _ Index = (*HNSW)(nil)
var _ Index = (*BruteForce)(nil)

func TestHNSWBasicSearch(t *testing.T) {
	h := NewHNSW(2, vector.Euclidean, DefaultHNSWConfig())
	h.Add("origin", vector.Vector{0, 0})
	h.Add("near", vector.Vector{1, 1})
	h.Add("far", vector.Vector{20, 20})

	res, err := h.Search(vector.Vector{0, 0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 || res[0].ID != "origin" || res[1].ID != "near" {
		t.Fatalf("unexpected result: %v", res)
	}
	if res[0].Score > res[1].Score {
		t.Fatalf("results not ascending: %v", res)
	}
}

func TestHNSWEmptyAndKBounds(t *testing.T) {
	h := NewHNSW(3, vector.Cosine, DefaultHNSWConfig())
	if res, _ := h.Search(vector.Vector{1, 0, 0}, 5); res != nil {
		t.Fatalf("empty index should return nil, got %v", res)
	}
	h.Add("a", vector.Vector{1, 0, 0})
	if res, _ := h.Search(vector.Vector{1, 0, 0}, 5); len(res) != 1 {
		t.Fatalf("got %d, want 1", len(res))
	}
}

func TestHNSWUpdateVector(t *testing.T) {
	h := NewHNSW(2, vector.Euclidean, DefaultHNSWConfig())
	h.Add("x", vector.Vector{0, 0})
	h.Add("x", vector.Vector{5, 5}) // same id, new vector
	if h.Len() != 1 {
		t.Fatalf("update created a duplicate: len=%d", h.Len())
	}
	res, _ := h.Search(vector.Vector{5, 5}, 1)
	if res[0].Score != 0 {
		t.Fatalf("vector was not updated, distance=%v", res[0].Score)
	}
}

// The headline property: HNSW should agree with the exact brute-force answer on the
// large majority of neighbors. Recall is the fraction of true top-k neighbors the
// approximate index also returns, averaged over many queries.
func TestHNSWRecallAgainstOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	const dim, n, k, queries = 32, 2000, 10, 200

	oracle := NewBruteForce(dim, vector.Euclidean)
	h := NewHNSW(dim, vector.Euclidean, DefaultHNSWConfig())
	for i := 0; i < n; i++ {
		v := randVec(rng, dim)
		id := fmt.Sprintf("v%d", i)
		oracle.Add(id, v)
		h.Add(id, v)
	}

	var totalRecall float64
	for q := 0; q < queries; q++ {
		query := randVec(rng, dim)
		want, _ := oracle.Search(query, k)
		got, _ := h.Search(query, k)

		truth := make(map[string]bool, k)
		for _, r := range want {
			truth[r.ID] = true
		}
		hits := 0
		for _, r := range got {
			if truth[r.ID] {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(k)
	}
	recall := totalRecall / float64(queries)
	t.Logf("HNSW recall@%d over %d queries: %.3f", k, queries, recall)
	if recall < 0.90 {
		t.Fatalf("recall %.3f below 0.90; index quality regressed", recall)
	}
}

// Raising efSearch should not lower recall: a wider search explores at least as much.
func TestHNSWEfSearchImprovesRecall(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	const dim, n, k, queries = 24, 1500, 10, 100

	oracle := NewBruteForce(dim, vector.Euclidean)
	h := NewHNSW(dim, vector.Euclidean, HNSWConfig{M: 8, EfConstruction: 100, EfSearch: 10})
	for i := 0; i < n; i++ {
		v := randVec(rng, dim)
		id := fmt.Sprintf("v%d", i)
		oracle.Add(id, v)
		h.Add(id, v)
	}
	measure := func() float64 {
		var total float64
		r := rand.New(rand.NewSource(5))
		for q := 0; q < queries; q++ {
			query := randVec(r, dim)
			want, _ := oracle.Search(query, k)
			got, _ := h.Search(query, k)
			truth := make(map[string]bool, k)
			for _, x := range want {
				truth[x.ID] = true
			}
			hits := 0
			for _, x := range got {
				if truth[x.ID] {
					hits++
				}
			}
			total += float64(hits) / float64(k)
		}
		return total / float64(queries)
	}

	low := measure()
	h.SetEfSearch(80)
	high := measure()
	t.Logf("recall ef=10: %.3f, ef=80: %.3f", low, high)
	if high < low-0.01 {
		t.Fatalf("higher efSearch lowered recall: %.3f -> %.3f", low, high)
	}
}
