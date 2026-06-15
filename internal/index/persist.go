package index

import (
	"encoding/gob"
	"io"
	"math/rand"
	"os"

	"github.com/qusay4421/vec-lite/internal/vector"
)

// hnswSnapshot is the on-disk form of an HNSW index. Building the graph is the
// expensive part (every insert runs a search), so persistence saves the finished
// graph structure, not just the raw vectors, and a reload skips the rebuild
// entirely. Stored as parallel arrays of exported fields so encoding/gob can handle
// it without custom codecs.
type hnswSnapshot struct {
	Dim            int
	Metric         int
	M              int
	M0             int
	EfConstruction int
	EfSearch       int
	ML             float64
	Entry          int
	TopLayer       int
	IDs            []string
	Vecs           [][]float32
	Neighbors      [][][]int
}

// Save writes the index to w. It takes a read lock, so it is safe to snapshot a live
// index that is still serving reads.
func (h *HNSW) Save(w io.Writer) error {
	h.mu.RLock()
	defer h.mu.RUnlock()

	snap := hnswSnapshot{
		Dim:            h.dim,
		Metric:         int(h.metric),
		M:              h.m,
		M0:             h.m0,
		EfConstruction: h.efConstruction,
		EfSearch:       h.efSearch,
		ML:             h.mL,
		Entry:          h.entry,
		TopLayer:       h.topLayer,
		IDs:            make([]string, len(h.nodes)),
		Vecs:           make([][]float32, len(h.nodes)),
		Neighbors:      make([][][]int, len(h.nodes)),
	}
	for i, nd := range h.nodes {
		snap.IDs[i] = nd.id
		snap.Vecs[i] = nd.vec
		snap.Neighbors[i] = nd.neighbors
	}
	return gob.NewEncoder(w).Encode(&snap)
}

// SaveFile writes the index to a path via a temp file and rename, so a crash mid-save
// cannot leave a half-written snapshot in place of a good one.
func (h *HNSW) SaveFile(path string) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := h.Save(f); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// LoadHNSW rebuilds an index from a snapshot. The id map and RNG are reconstructed
// (the RNG only affects the level of future inserts, not existing structure).
func LoadHNSW(r io.Reader) (*HNSW, error) {
	var snap hnswSnapshot
	if err := gob.NewDecoder(r).Decode(&snap); err != nil {
		return nil, err
	}
	h := &HNSW{
		metric:         vector.Metric(snap.Metric),
		dim:            snap.Dim,
		m:              snap.M,
		m0:             snap.M0,
		efConstruction: snap.EfConstruction,
		efSearch:       snap.EfSearch,
		mL:             snap.ML,
		rng:            rand.New(rand.NewSource(1)),
		byID:           make(map[string]int, len(snap.IDs)),
		entry:          snap.Entry,
		topLayer:       snap.TopLayer,
		nodes:          make([]hnswNode, len(snap.IDs)),
	}
	for i := range snap.IDs {
		h.nodes[i] = hnswNode{
			id:        snap.IDs[i],
			vec:       vector.Vector(snap.Vecs[i]),
			neighbors: snap.Neighbors[i],
		}
		h.byID[snap.IDs[i]] = i
	}
	return h, nil
}

func LoadHNSWFile(path string) (*HNSW, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return LoadHNSW(f)
}
