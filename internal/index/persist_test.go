package index

import (
	"bytes"
	"fmt"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/qusay4421/vec-lite/internal/vector"
)

// A reloaded index must be identical to the original: same size and the same search
// results for the same queries, with no rebuild.
func TestHNSWSaveLoadRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const dim, n, k = 24, 800, 10
	orig := NewHNSW(dim, vector.Cosine, DefaultHNSWConfig())
	queries := make([]vector.Vector, 20)
	for i := 0; i < n; i++ {
		orig.Add(fmt.Sprintf("v%d", i), randVec(rng, dim))
	}
	for i := range queries {
		queries[i] = randVec(rng, dim)
	}

	var buf bytes.Buffer
	if err := orig.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadHNSW(&buf)
	if err != nil {
		t.Fatal(err)
	}

	if loaded.Len() != orig.Len() {
		t.Fatalf("len after load = %d, want %d", loaded.Len(), orig.Len())
	}
	for _, q := range queries {
		want, _ := orig.Search(q, k)
		got, _ := loaded.Search(q, k)
		if len(want) != len(got) {
			t.Fatalf("result count differs: %d vs %d", len(want), len(got))
		}
		for i := range want {
			if want[i].ID != got[i].ID {
				t.Fatalf("result %d differs after reload: %s vs %s", i, want[i].ID, got[i].ID)
			}
		}
	}
}

// A loaded index must still accept new inserts and keep serving correctly.
func TestHNSWLoadThenInsert(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	dim := 16
	orig := NewHNSW(dim, vector.Euclidean, DefaultHNSWConfig())
	for i := 0; i < 100; i++ {
		orig.Add(fmt.Sprintf("v%d", i), randVec(rng, dim))
	}
	path := filepath.Join(t.TempDir(), "index.idx")
	if err := orig.SaveFile(path); err != nil {
		t.Fatal(err)
	}

	loaded, err := LoadHNSWFile(path)
	if err != nil {
		t.Fatal(err)
	}
	probe := randVec(rng, dim)
	loaded.Add("probe", probe)
	res, _ := loaded.Search(probe, 1)
	if len(res) != 1 || res[0].ID != "probe" || res[0].Score > 1e-5 {
		t.Fatalf("inserted vector not found after load: %v", res)
	}
}
