package index

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"

	"github.com/qusay4421/vec-lite/internal/vector"
)

func TestSearchReturnsNearestInOrder(t *testing.T) {
	idx := NewBruteForce(2, vector.Euclidean)
	idx.Add("origin", vector.Vector{0, 0})
	idx.Add("near", vector.Vector{1, 1})
	idx.Add("far", vector.Vector{9, 9})

	res, err := idx.Search(vector.Vector{0, 0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 {
		t.Fatalf("got %d results, want 2", len(res))
	}
	if res[0].ID != "origin" || res[1].ID != "near" {
		t.Fatalf("wrong order: %v", res)
	}
	// Results must be sorted by ascending distance.
	if res[0].Score > res[1].Score {
		t.Fatalf("results not ascending: %v", res)
	}
}

func TestSearchKLargerThanN(t *testing.T) {
	idx := NewBruteForce(3, vector.Cosine)
	idx.Add("a", vector.Vector{1, 0, 0})
	idx.Add("b", vector.Vector{0, 1, 0})
	res, _ := idx.Search(vector.Vector{1, 0, 0}, 10)
	if len(res) != 2 {
		t.Fatalf("got %d, want all 2 stored", len(res))
	}
}

func TestDimMismatch(t *testing.T) {
	idx := NewBruteForce(4, vector.Euclidean)
	if err := idx.Add("x", vector.Vector{1, 2, 3}); err != ErrDim {
		t.Fatalf("Add err = %v, want ErrDim", err)
	}
	if _, err := idx.Search(vector.Vector{1, 2, 3}, 1); err != ErrDim {
		t.Fatalf("Search err = %v, want ErrDim", err)
	}
}

// The heap-based top-k must agree with an independent full-sort reference on random
// data. This is the property that lets BruteForce serve as the oracle for the
// approximate index later.
func TestMatchesNaiveReference(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const dim, n, k = 16, 500, 10
	idx := NewBruteForce(dim, vector.Euclidean)

	vecs := make([]vector.Vector, n)
	for i := 0; i < n; i++ {
		v := randVec(rng, dim)
		vecs[i] = v
		idx.Add(fmt.Sprintf("v%d", i), v)
	}
	query := randVec(rng, dim)

	got, err := idx.Search(query, k)
	if err != nil {
		t.Fatal(err)
	}

	// Independent reference: distance to every vector, fully sorted.
	type pair struct {
		id string
		d  float32
	}
	ref := make([]pair, n)
	for i, v := range vecs {
		d, _ := vector.Euclidean.Distance(query, v)
		ref[i] = pair{fmt.Sprintf("v%d", i), d}
	}
	sort.Slice(ref, func(i, j int) bool { return ref[i].d < ref[j].d })

	for i := 0; i < k; i++ {
		if got[i].ID != ref[i].id {
			t.Fatalf("rank %d: got %s (%.4f), want %s (%.4f)", i, got[i].ID, got[i].Score, ref[i].id, ref[i].d)
		}
	}
}

func randVec(rng *rand.Rand, dim int) vector.Vector {
	v := make(vector.Vector, dim)
	for i := range v {
		v[i] = rng.Float32()
	}
	return v
}
