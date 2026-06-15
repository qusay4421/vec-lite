// Command server exposes a vec-lite collection over HTTP/JSON: insert vectors with
// metadata, and run nearest-neighbor queries with an optional metadata filter.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"log"
	"net/http"
	"strings"

	"github.com/qusay4421/vec-lite/internal/collection"
	"github.com/qusay4421/vec-lite/internal/index"
	"github.com/qusay4421/vec-lite/internal/vector"
)

type addRequest struct {
	Vector   []float32         `json:"vector"`
	Metadata map[string]string `json:"metadata"`
}

type searchRequest struct {
	Vector []float32         `json:"vector"`
	K      int               `json:"k"`
	Filter map[string]string `json:"filter"`
}

type hit struct {
	ID       string            `json:"id"`
	Score    float32           `json:"score"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

func main() {
	addr := flag.String("addr", ":8081", "HTTP listen address")
	dim := flag.Int("dim", 128, "vector dimension")
	metricName := flag.String("metric", "cosine", "distance metric: cosine, euclidean, inner_product")
	flag.Parse()

	metric, err := parseMetric(*metricName)
	if err != nil {
		log.Fatal(err)
	}
	col := collection.New(*dim, metric, index.DefaultHNSWConfig())

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })

	// PUT /vectors/{id} inserts or updates a vector and its metadata.
	mux.HandleFunc("/vectors/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut && r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		id := strings.TrimPrefix(r.URL.Path, "/vectors/")
		if id == "" {
			http.Error(w, "missing id", http.StatusBadRequest)
			return
		}
		var req addRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
			return
		}
		if len(req.Vector) != *dim {
			http.Error(w, "vector dimension mismatch", http.StatusBadRequest)
			return
		}
		if err := col.Add(id, vector.Vector(req.Vector), req.Metadata); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	// POST /search runs a query, applying the metadata filter when present.
	mux.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req searchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad json: "+err.Error(), http.StatusBadRequest)
			return
		}
		if len(req.Vector) != *dim {
			http.Error(w, "vector dimension mismatch", http.StatusBadRequest)
			return
		}
		if req.K <= 0 {
			req.K = 10
		}
		var hits []collection.Hit
		if len(req.Filter) > 0 {
			hits, err = col.SearchFiltered(vector.Vector(req.Vector), req.K, filterFrom(req.Filter))
		} else {
			hits, err = col.Search(vector.Vector(req.Vector), req.K)
		}
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		out := make([]hit, len(hits))
		for i, h := range hits {
			out[i] = hit{ID: h.ID, Score: h.Score, Metadata: h.Metadata}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(out)
	})

	log.Printf("vec-lite server on %s (dim %d, metric %s)", *addr, *dim, metric)
	if err := http.ListenAndServe(*addr, mux); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

// filterFrom turns a key/value map into an AND of equality predicates.
func filterFrom(m map[string]string) collection.Predicate {
	preds := make([]collection.Predicate, 0, len(m))
	for k, v := range m {
		preds = append(preds, collection.Equals(k, v))
	}
	return collection.And(preds...)
}

func parseMetric(name string) (vector.Metric, error) {
	switch name {
	case "cosine":
		return vector.Cosine, nil
	case "euclidean":
		return vector.Euclidean, nil
	case "inner_product":
		return vector.InnerProduct, nil
	default:
		return 0, errors.New("unknown metric: " + name)
	}
}
