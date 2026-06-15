// Command bench measures the core tradeoff of the vector index: how recall and query
// latency move as efSearch changes, against the exact brute-force answer. It also
// reports build time and the speedup over a full scan. Standard library only.
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"sort"
	"time"

	"github.com/qusay4421/vec-lite/internal/index"
	"github.com/qusay4421/vec-lite/internal/vector"
)

func main() {
	dim := flag.Int("dim", 128, "vector dimension")
	n := flag.Int("n", 20000, "number of vectors")
	queries := flag.Int("queries", 1000, "number of query vectors")
	k := flag.Int("k", 10, "neighbors per query")
	flag.Parse()

	rng := rand.New(rand.NewSource(42))
	data := make([]vector.Vector, *n)
	for i := range data {
		data[i] = randVec(rng, *dim)
	}
	qs := make([]vector.Vector, *queries)
	for i := range qs {
		qs[i] = randVec(rng, *dim)
	}

	// Exact oracle for both recall ground truth and the brute-force latency baseline.
	bf := index.NewBruteForce(*dim, vector.Euclidean)
	for i, v := range data {
		bf.Add(fmt.Sprintf("v%d", i), v)
	}

	h := index.NewHNSW(*dim, vector.Euclidean, index.DefaultHNSWConfig())
	buildStart := time.Now()
	for i, v := range data {
		h.Add(fmt.Sprintf("v%d", i), v)
	}
	buildTime := time.Since(buildStart)

	truth := make([]map[string]bool, *queries)
	bfLat := make([]time.Duration, *queries)
	for i, q := range qs {
		t0 := time.Now()
		res, _ := bf.Search(q, *k)
		bfLat[i] = time.Since(t0)
		m := make(map[string]bool, *k)
		for _, r := range res {
			m[r.ID] = true
		}
		truth[i] = m
	}
	bfP50 := percentile(bfLat, 0.50)

	fmt.Printf("dataset: %d vectors, dim %d, k %d, %d queries\n", *n, *dim, *k, *queries)
	fmt.Printf("HNSW build: %s (%.0f inserts/sec)\n", buildTime.Round(time.Millisecond), float64(*n)/buildTime.Seconds())
	fmt.Printf("brute-force query p50: %s\n\n", bfP50.Round(time.Microsecond))

	fmt.Printf("%-9s %-8s %-12s %-12s %-9s\n", "efSearch", "recall", "p50", "p99", "speedup")
	for _, ef := range []int{10, 25, 50, 100, 200} {
		h.SetEfSearch(ef)
		lat := make([]time.Duration, *queries)
		var recall float64
		for i, q := range qs {
			t0 := time.Now()
			res, _ := h.Search(q, *k)
			lat[i] = time.Since(t0)
			hits := 0
			for _, r := range res {
				if truth[i][r.ID] {
					hits++
				}
			}
			recall += float64(hits) / float64(*k)
		}
		recall /= float64(*queries)
		p50 := percentile(lat, 0.50)
		p99 := percentile(lat, 0.99)
		speedup := float64(bfP50) / float64(p50)
		fmt.Printf("%-9d %-8.3f %-12s %-12s %-8.1fx\n",
			ef, recall, p50.Round(time.Microsecond), p99.Round(time.Microsecond), speedup)
	}
}

func percentile(ds []time.Duration, p float64) time.Duration {
	c := append([]time.Duration(nil), ds...)
	sort.Slice(c, func(i, j int) bool { return c[i] < c[j] })
	idx := int(float64(len(c))*p) - 1
	if idx < 0 {
		idx = 0
	}
	return c[idx]
}

func randVec(rng *rand.Rand, dim int) vector.Vector {
	v := make(vector.Vector, dim)
	for i := range v {
		v[i] = rng.Float32()
	}
	return v
}
