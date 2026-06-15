// Package collection adds metadata and filtered search on top of a vector index.
// A vector index alone answers "nearest neighbors"; real use needs "nearest
// neighbors that also match this filter" (a price range, a category, a tenant).
package collection

import (
	"sync"

	"github.com/qusay4421/vec-lite/internal/index"
	"github.com/qusay4421/vec-lite/internal/vector"
)

// Metadata is a small string-keyed payload attached to each vector.
type Metadata map[string]string

// Predicate decides whether a record qualifies for a filtered search.
type Predicate func(Metadata) bool

// Equals matches records whose key holds value. Composable building block for
// filters; And combines several.
func Equals(key, value string) Predicate {
	return func(m Metadata) bool { return m[key] == value }
}

func And(preds ...Predicate) Predicate {
	return func(m Metadata) bool {
		for _, p := range preds {
			if !p(m) {
				return false
			}
		}
		return true
	}
}

// Hit is a search result with its payload attached.
type Hit struct {
	ID       string
	Score    float32
	Metadata Metadata
}

type Collection struct {
	mu   sync.RWMutex
	idx  *index.HNSW
	meta map[string]Metadata
}

func New(dim int, metric vector.Metric, cfg index.HNSWConfig) *Collection {
	return &Collection{
		idx:  index.NewHNSW(dim, metric, cfg),
		meta: make(map[string]Metadata),
	}
}

func (c *Collection) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.meta)
}

func (c *Collection) Add(id string, v vector.Vector, md Metadata) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.idx.Add(id, v); err != nil {
		return err
	}
	cp := make(Metadata, len(md))
	for k, val := range md {
		cp[k] = val
	}
	c.meta[id] = cp
	return nil
}

func (c *Collection) Search(query vector.Vector, k int) ([]Hit, error) {
	res, err := c.idx.Search(query, k)
	if err != nil {
		return nil, err
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.attach(res, k), nil
}

// overfetchFactor is how many extra candidates SearchFiltered pulls from the index
// per requested result. The nearest k overall may contain few that pass a filter, so
// the index is asked for more and the surplus is filtered down.
const overfetchFactor = 8

// SearchFiltered returns the k nearest records whose metadata satisfies pred.
//
// It uses post-filtering with over-fetch: ask the index for k*factor neighbors, then
// keep the first k that pass. This is simple and keeps the fast graph search, but a
// highly selective filter can still leave fewer than k matches inside the fetched
// window, in which case it under-returns. The exact-but-slower alternative is to scan
// the filtered subset directly; see DESIGN.md.
func (c *Collection) SearchFiltered(query vector.Vector, k int, pred Predicate) ([]Hit, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	fetch := k * overfetchFactor
	if n := c.idx.Len(); fetch > n {
		fetch = n
	}
	res, err := c.idx.Search(query, fetch)
	if err != nil {
		return nil, err
	}
	out := make([]Hit, 0, k)
	for _, r := range res {
		md := c.meta[r.ID]
		if pred(md) {
			out = append(out, Hit{ID: r.ID, Score: r.Score, Metadata: md})
			if len(out) == k {
				break
			}
		}
	}
	return out, nil
}

func (c *Collection) attach(res []index.Result, k int) []Hit {
	out := make([]Hit, len(res))
	for i, r := range res {
		out[i] = Hit{ID: r.ID, Score: r.Score, Metadata: c.meta[r.ID]}
	}
	return out
}
