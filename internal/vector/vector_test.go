package vector

import (
	"math"
	"testing"
)

func approx(a, b float32) bool {
	return math.Abs(float64(a-b)) < 1e-5
}

func TestEuclidean(t *testing.T) {
	d, err := Euclidean.Distance(Vector{0, 0}, Vector{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if !approx(d, 5) { // classic 3-4-5
		t.Fatalf("euclidean = %v, want 5", d)
	}
}

func TestCosineIdenticalAndOrthogonal(t *testing.T) {
	// Same direction: cosine similarity 1, so distance 0.
	if d, _ := Cosine.Distance(Vector{1, 1}, Vector{2, 2}); !approx(d, 0) {
		t.Fatalf("cosine of parallel vectors = %v, want 0", d)
	}
	// Orthogonal: similarity 0, so distance 1.
	if d, _ := Cosine.Distance(Vector{1, 0}, Vector{0, 1}); !approx(d, 1) {
		t.Fatalf("cosine of orthogonal vectors = %v, want 1", d)
	}
	// Opposite: similarity -1, so distance 2.
	if d, _ := Cosine.Distance(Vector{1, 0}, Vector{-1, 0}); !approx(d, 2) {
		t.Fatalf("cosine of opposite vectors = %v, want 2", d)
	}
}

func TestCosineZeroVector(t *testing.T) {
	if d, _ := Cosine.Distance(Vector{0, 0}, Vector{1, 1}); !approx(d, 1) {
		t.Fatalf("cosine with zero vector = %v, want 1 (undefined treated as far)", d)
	}
}

func TestInnerProductRanksByDot(t *testing.T) {
	// Larger dot product must mean smaller distance.
	near, _ := InnerProduct.Distance(Vector{1, 1}, Vector{2, 2}) // dot 4 -> -4
	far, _ := InnerProduct.Distance(Vector{1, 1}, Vector{1, 0})  // dot 1 -> -1
	if !(near < far) {
		t.Fatalf("inner product ranking wrong: near=%v far=%v", near, far)
	}
}

func TestDimMismatch(t *testing.T) {
	if _, err := Euclidean.Distance(Vector{1, 2}, Vector{1}); err != ErrDim {
		t.Fatalf("err = %v, want ErrDim", err)
	}
}

func TestNormalize(t *testing.T) {
	u := Normalize(Vector{3, 4})
	if !approx(norm(u), 1) {
		t.Fatalf("normalized norm = %v, want 1", norm(u))
	}
	// A normalized vector's cosine distance to its source is 0 (same direction).
	if d, _ := Cosine.Distance(u, Vector{3, 4}); !approx(d, 0) {
		t.Fatalf("cosine(normalized, original) = %v, want 0", d)
	}
}
