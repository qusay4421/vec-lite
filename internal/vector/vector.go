// Package vector holds fixed-dimension float32 vectors and the distance metrics
// used to compare them.
//
// Every metric follows a "lower means closer" convention. Cosine and inner product
// are naturally similarities (higher means closer), so they are returned as
// distances (1 - sim, and -sim) instead. This lets any index rank candidates the
// same way regardless of metric: smallest distance wins.
package vector

import (
	"errors"
	"math"
)

// Vector is a dense embedding. float32 (not float64) halves memory and bandwidth,
// which matters because a vector index is dominated by the size of the raw vectors.
type Vector []float32

// ErrDim means two vectors that must match in length did not.
var ErrDim = errors.New("vector: dimension mismatch")

// Metric selects how distance is measured.
type Metric int

const (
	Euclidean Metric = iota // L2 distance
	Cosine                  // 1 - cosine similarity
	InnerProduct            // negative dot product (for maximum inner-product search)
)

func (m Metric) String() string {
	switch m {
	case Euclidean:
		return "euclidean"
	case Cosine:
		return "cosine"
	case InnerProduct:
		return "inner_product"
	default:
		return "unknown"
	}
}

// Distance returns the distance between a and b under the metric, where smaller is
// more similar.
func (m Metric) Distance(a, b Vector) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDim
	}
	switch m {
	case Euclidean:
		return euclidean(a, b), nil
	case Cosine:
		return cosineDistance(a, b), nil
	case InnerProduct:
		return -dot(a, b), nil
	default:
		return 0, errors.New("vector: unknown metric")
	}
}

func euclidean(a, b Vector) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	// The square root keeps the value a true L2 distance. Ranking alone would not
	// need it (it is monotonic), but returning real distances keeps the API honest.
	return float32(math.Sqrt(float64(sum)))
}

func cosineDistance(a, b Vector) float32 {
	d := dot(a, b)
	na, nb := norm(a), norm(b)
	// A zero-length vector has no direction, so similarity is undefined. Treat it as
	// maximally dissimilar (distance 1) rather than dividing by zero.
	if na == 0 || nb == 0 {
		return 1
	}
	sim := d / (na * nb)
	// Floating-point error can push the cosine just past the unit interval; clamp so
	// the returned distance stays within [0, 2].
	if sim > 1 {
		sim = 1
	} else if sim < -1 {
		sim = -1
	}
	return 1 - sim
}

func dot(a, b Vector) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func norm(a Vector) float32 {
	return float32(math.Sqrt(float64(dot(a, a))))
}

// Normalize returns a unit-length copy of v. With normalized vectors, cosine
// distance reduces to a cheaper inner product, an optimization indexes can use.
func Normalize(v Vector) Vector {
	n := norm(v)
	out := make(Vector, len(v))
	if n == 0 {
		return out
	}
	for i := range v {
		out[i] = v[i] / n
	}
	return out
}
