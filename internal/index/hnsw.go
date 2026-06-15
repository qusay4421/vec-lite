package index

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/qusay4421/vec-lite/internal/vector"
)

// HNSW is an approximate nearest-neighbor index built on a Hierarchical Navigable
// Small World graph. Nodes live on multiple layers: the top layers are sparse and
// act as an express lane, the bottom layer holds every node. A search greedily hops
// toward the query at the top, then descends and refines. This gives roughly
// logarithmic search instead of the brute-force linear scan, at the cost of
// occasionally missing a true neighbor (measured as recall in the benchmarks).
type HNSW struct {
	mu     sync.RWMutex
	metric vector.Metric
	dim    int

	m              int     // target neighbors per node on upper layers
	m0             int     // target neighbors on layer 0 (denser, conventionally 2*m)
	efConstruction int     // candidate-list width while building
	efSearch       int     // candidate-list width while querying
	mL             float64 // level-generation scale, 1/ln(m)
	rng            *rand.Rand

	nodes    []hnswNode
	byID     map[string]int
	entry    int // entry-point node index, -1 when empty
	topLayer int
}

type hnswNode struct {
	id        string
	vec       vector.Vector
	neighbors [][]int // neighbors[layer] = node indices linked at that layer
}

// HNSWConfig holds the tuning knobs. Larger values raise recall and memory or build
// time. The defaults are common starting points for datasets up to ~1M vectors.
type HNSWConfig struct {
	M              int
	EfConstruction int
	EfSearch       int
}

func DefaultHNSWConfig() HNSWConfig {
	return HNSWConfig{M: 16, EfConstruction: 200, EfSearch: 50}
}

func NewHNSW(dim int, m vector.Metric, cfg HNSWConfig) *HNSW {
	if cfg.M < 2 {
		cfg.M = 2
	}
	if cfg.EfConstruction < cfg.M {
		cfg.EfConstruction = cfg.M
	}
	if cfg.EfSearch < 1 {
		cfg.EfSearch = 1
	}
	return &HNSW{
		metric:         m,
		dim:            dim,
		m:              cfg.M,
		m0:             cfg.M * 2,
		efConstruction: cfg.EfConstruction,
		efSearch:       cfg.EfSearch,
		mL:             1 / math.Log(float64(cfg.M)),
		rng:            rand.New(rand.NewSource(1)),
		byID:           make(map[string]int),
		entry:          -1,
	}
}

// SetEfSearch tunes the query-time accuracy/speed tradeoff without rebuilding. The
// benchmark day sweeps this to plot recall against latency.
func (h *HNSW) SetEfSearch(ef int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if ef < 1 {
		ef = 1
	}
	h.efSearch = ef
}

func (h *HNSW) Len() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.nodes)
}

func (h *HNSW) Add(id string, v vector.Vector) error {
	if len(v) != h.dim {
		return ErrDim
	}
	h.mu.Lock()
	defer h.mu.Unlock()

	// Updating an existing id replaces its vector but keeps its graph links. A full
	// relink on update is more correct and is noted as future work in DESIGN.md.
	if existing, ok := h.byID[id]; ok {
		h.nodes[existing].vec = append(vector.Vector(nil), v...)
		return nil
	}

	level := h.randomLevel()
	idx := len(h.nodes)
	nd := hnswNode{
		id:        id,
		vec:       append(vector.Vector(nil), v...),
		neighbors: make([][]int, level+1),
	}
	h.nodes = append(h.nodes, nd)
	h.byID[id] = idx

	// First node becomes the entry point and we are done.
	if h.entry == -1 {
		h.entry = idx
		h.topLayer = level
		return nil
	}

	ep := h.entry
	// Descend the express lanes above the new node's top level with a width-1 greedy
	// search, narrowing the entry point before we start linking.
	for l := h.topLayer; l > level; l-- {
		if res := h.searchLayer(v, []int{ep}, 1, l); len(res) > 0 {
			ep = res[0].idx
		}
	}

	entryPoints := []int{ep}
	start := level
	if h.topLayer < start {
		start = h.topLayer
	}
	for l := start; l >= 0; l-- {
		res := h.searchLayer(v, entryPoints, h.efConstruction, l)
		mMax := h.m
		if l == 0 {
			mMax = h.m0
		}
		selected := selectClosest(res, mMax)
		h.nodes[idx].neighbors[l] = selected

		// Links are bidirectional; after adding the back-link, prune the neighbor's
		// list so no node exceeds the per-layer degree cap.
		for _, nb := range selected {
			h.nodes[nb].neighbors[l] = append(h.nodes[nb].neighbors[l], idx)
			if len(h.nodes[nb].neighbors[l]) > mMax {
				h.nodes[nb].neighbors[l] = h.pruneNeighbors(nb, l, mMax)
			}
		}

		// Carry this layer's candidates down as entry points for the next.
		entryPoints = entryPoints[:0]
		for _, c := range res {
			entryPoints = append(entryPoints, c.idx)
		}
	}

	if level > h.topLayer {
		h.topLayer = level
		h.entry = idx
	}
	return nil
}

func (h *HNSW) Search(query vector.Vector, k int) ([]Result, error) {
	if len(query) != h.dim {
		return nil, ErrDim
	}
	if k <= 0 {
		return nil, nil
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.entry == -1 {
		return nil, nil
	}

	ep := h.entry
	for l := h.topLayer; l > 0; l-- {
		if res := h.searchLayer(query, []int{ep}, 1, l); len(res) > 0 {
			ep = res[0].idx
		}
	}
	ef := h.efSearch
	if ef < k {
		ef = k // must explore at least k to return k
	}
	res := h.searchLayer(query, []int{ep}, ef, 0)
	if len(res) > k {
		res = res[:k]
	}
	out := make([]Result, len(res))
	for i, c := range res {
		out[i] = Result{ID: h.nodes[c.idx].id, Score: c.dist}
	}
	return out, nil
}

// searchLayer runs the greedy best-first search at one layer: it expands the closest
// unexplored candidate until the frontier can no longer improve the ef-sized result
// set. Returns up to ef nodes sorted nearest first.
func (h *HNSW) searchLayer(query vector.Vector, entryPoints []int, ef, layer int) []candidate {
	visited := make(map[int]struct{}, ef*2)
	cand := &minHeap{}  // frontier to expand, closest first
	res := &maxHeapC{} // best ef found, farthest on top for cheap eviction

	for _, ep := range entryPoints {
		d := h.distTo(query, ep)
		visited[ep] = struct{}{}
		heap.Push(cand, candidate{ep, d})
		heap.Push(res, candidate{ep, d})
	}

	for cand.Len() > 0 {
		c := heap.Pop(cand).(candidate)
		// Once the nearest frontier node is farther than the current worst result and
		// the result set is full, nothing left can improve it.
		if res.Len() >= ef && c.dist > (*res)[0].dist {
			break
		}
		if layer >= len(h.nodes[c.idx].neighbors) {
			continue
		}
		for _, nb := range h.nodes[c.idx].neighbors[layer] {
			if _, seen := visited[nb]; seen {
				continue
			}
			visited[nb] = struct{}{}
			d := h.distTo(query, nb)
			if res.Len() < ef || d < (*res)[0].dist {
				heap.Push(cand, candidate{nb, d})
				heap.Push(res, candidate{nb, d})
				if res.Len() > ef {
					heap.Pop(res)
				}
			}
		}
	}

	out := make([]candidate, res.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(res).(candidate)
	}
	return out
}

// pruneNeighbors keeps only the m closest of a node's current neighbors at a layer,
// recomputing distances from that node. This bounds degree after a back-link push.
func (h *HNSW) pruneNeighbors(nodeIdx, layer, m int) []int {
	cur := h.nodes[nodeIdx].neighbors[layer]
	cands := make([]candidate, len(cur))
	for i, nb := range cur {
		cands[i] = candidate{nb, h.distBetween(nodeIdx, nb)}
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].dist < cands[j].dist })
	out := make([]int, m)
	for i := 0; i < m; i++ {
		out[i] = cands[i].idx
	}
	return out
}

// randomLevel draws an exponentially decaying layer, so each higher layer holds
// roughly 1/m of the layer below it.
func (h *HNSW) randomLevel() int {
	r := h.rng.Float64()
	if r <= 0 {
		r = 1e-12 // guard against log(0)
	}
	return int(-math.Log(r) * h.mL)
}

func (h *HNSW) distTo(query vector.Vector, idx int) float32 {
	d, _ := h.metric.Distance(query, h.nodes[idx].vec)
	return d
}

func (h *HNSW) distBetween(a, b int) float32 {
	d, _ := h.metric.Distance(h.nodes[a].vec, h.nodes[b].vec)
	return d
}

// selectClosest takes the m nearest of an already-ascending candidate list.
func selectClosest(cands []candidate, m int) []int {
	if len(cands) > m {
		cands = cands[:m]
	}
	out := make([]int, len(cands))
	for i, c := range cands {
		out[i] = c.idx
	}
	return out
}

type candidate struct {
	idx  int
	dist float32
}

// minHeap orders closest-first, for the expansion frontier.
type minHeap []candidate

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	c := old[n-1]
	*h = old[:n-1]
	return c
}

// maxHeapC orders farthest-first, so the result set's worst entry is the root and
// cheap to evict.
type maxHeapC []candidate

func (h maxHeapC) Len() int           { return len(h) }
func (h maxHeapC) Less(i, j int) bool { return h[i].dist > h[j].dist }
func (h maxHeapC) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxHeapC) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *maxHeapC) Pop() any {
	old := *h
	n := len(old)
	c := old[n-1]
	*h = old[:n-1]
	return c
}
