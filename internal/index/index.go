// Package index provides vector search indexes over the vector package.
//
// BruteForce is an exact, linear-scan index. It is both a usable index for small
// data and the correctness oracle the approximate index (HNSW, added later) is
// tested against: an approximate search is judged by how often it agrees with the
// exact answer brute force produces.
package index

import (
	"container/heap"
	"errors"
	"sync"

	"github.com/qusay4421/vec-lite/internal/vector"
)

// Result is one search hit. Score is the metric distance, so lower is closer.
type Result struct {
	ID    string
	Score float32
}

// Index is the behavior every vector index shares, so callers and tests can treat
// brute force and the approximate index interchangeably.
type Index interface {
	Add(id string, v vector.Vector) error
	Search(query vector.Vector, k int) ([]Result, error)
	Len() int
}

var ErrDim = errors.New("index: vector dimension does not match index")

// BruteForce scans every stored vector on each query. O(n) per search, exact by
// construction.
type BruteForce struct {
	mu     sync.RWMutex
	metric vector.Metric
	dim    int
	ids    []string
	vecs   []vector.Vector
}

func NewBruteForce(dim int, m vector.Metric) *BruteForce {
	return &BruteForce{metric: m, dim: dim}
}

func (b *BruteForce) Add(id string, v vector.Vector) error {
	if len(v) != b.dim {
		return ErrDim
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	// Copy so a caller reusing its slice cannot mutate indexed data later.
	cp := make(vector.Vector, len(v))
	copy(cp, v)
	b.ids = append(b.ids, id)
	b.vecs = append(b.vecs, cp)
	return nil
}

func (b *BruteForce) Len() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.ids)
}

// Search returns the k closest vectors to query, nearest first. It keeps a bounded
// max-heap of the best k seen so the cost is O(n log k) and O(k) memory rather than
// sorting all n distances.
func (b *BruteForce) Search(query vector.Vector, k int) ([]Result, error) {
	if len(query) != b.dim {
		return nil, ErrDim
	}
	if k <= 0 {
		return nil, nil
	}
	b.mu.RLock()
	defer b.mu.RUnlock()

	h := &maxHeap{}
	for i := range b.vecs {
		d, err := b.metric.Distance(query, b.vecs[i])
		if err != nil {
			return nil, err
		}
		r := Result{ID: b.ids[i], Score: d}
		if h.Len() < k {
			heap.Push(h, r)
		} else if d < (*h)[0].Score {
			// Heap root is the current worst of the best k; replace it when we find
			// something closer.
			(*h)[0] = r
			heap.Fix(h, 0)
		}
	}

	// Pop the max repeatedly to fill the output from the back, yielding ascending
	// distance (nearest first).
	out := make([]Result, h.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(h).(Result)
	}
	return out, nil
}

// maxHeap orders by largest distance at the root, so the farthest of the current
// top-k is the one evicted when a closer candidate arrives.
type maxHeap []Result

func (h maxHeap) Len() int            { return len(h) }
func (h maxHeap) Less(i, j int) bool  { return h[i].Score > h[j].Score }
func (h maxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)         { *h = append(*h, x.(Result)) }
func (h *maxHeap) Pop() any {
	old := *h
	n := len(old)
	r := old[n-1]
	*h = old[:n-1]
	return r
}
