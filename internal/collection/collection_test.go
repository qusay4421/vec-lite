package collection

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/qusay4421/vec-lite/internal/index"
	"github.com/qusay4421/vec-lite/internal/vector"
)

func randVec(rng *rand.Rand, dim int) vector.Vector {
	v := make(vector.Vector, dim)
	for i := range v {
		v[i] = rng.Float32()
	}
	return v
}

// Filtered search must only return records that pass the predicate, in nearest order.
func TestFilteredSearchRespectsPredicate(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	dim := 16
	c := New(dim, vector.Euclidean, index.DefaultHNSWConfig())
	for i := 0; i < 600; i++ {
		cat := []string{"a", "b", "c"}[i%3]
		c.Add(fmt.Sprintf("v%d", i), randVec(rng, dim), Metadata{"category": cat})
	}

	hits, err := c.SearchFiltered(randVec(rng, dim), 10, Equals("category", "a"))
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 {
		t.Fatal("expected some matches")
	}
	for i, h := range hits {
		if h.Metadata["category"] != "a" {
			t.Fatalf("hit %s has category %s, want a", h.ID, h.Metadata["category"])
		}
		if i > 0 && hits[i-1].Score > h.Score {
			t.Fatalf("results not ascending at %d", i)
		}
	}
}

// And-composed filters must require every clause.
func TestAndFilter(t *testing.T) {
	dim := 4
	c := New(dim, vector.Euclidean, index.DefaultHNSWConfig())
	c.Add("x", vector.Vector{1, 0, 0, 0}, Metadata{"category": "a", "tier": "pro"})
	c.Add("y", vector.Vector{1, 0, 0, 0}, Metadata{"category": "a", "tier": "free"})

	hits, _ := c.SearchFiltered(vector.Vector{1, 0, 0, 0}, 5,
		And(Equals("category", "a"), Equals("tier", "pro")))
	if len(hits) != 1 || hits[0].ID != "x" {
		t.Fatalf("And filter wrong: %v", hits)
	}
}

// Over-fetch post-filtering should recover most of the true filtered neighbors that
// an exact scan over the matching subset would return.
func TestFilteredSearchRecallVsExactSubset(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	dim, n, k := 24, 1500, 10
	c := New(dim, vector.Euclidean, index.DefaultHNSWConfig())

	// Keep the matching subset's vectors so we can compute the exact filtered answer.
	exact := index.NewBruteForce(dim, vector.Euclidean)
	for i := 0; i < n; i++ {
		v := randVec(rng, dim)
		cat := "other"
		if i%4 == 0 {
			cat = "target"
		}
		id := fmt.Sprintf("v%d", i)
		c.Add(id, v, Metadata{"category": cat})
		if cat == "target" {
			exact.Add(id, v)
		}
	}

	var total float64
	const queries = 100
	for q := 0; q < queries; q++ {
		query := randVec(rng, dim)
		want, _ := exact.Search(query, k)
		got, _ := c.SearchFiltered(query, k, Equals("category", "target"))

		truth := make(map[string]bool, len(want))
		for _, r := range want {
			truth[r.ID] = true
		}
		hits := 0
		for _, h := range got {
			if truth[h.ID] {
				hits++
			}
		}
		total += float64(hits) / float64(len(want))
	}
	recall := total / float64(queries)
	t.Logf("filtered recall@%d vs exact subset: %.3f", k, recall)
	if recall < 0.85 {
		t.Fatalf("filtered recall %.3f below 0.85", recall)
	}
}
