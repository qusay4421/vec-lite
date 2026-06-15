package embed

import (
	"testing"

	"github.com/qusay4421/vec-lite/internal/vector"
)

func TestEmbedDimAndNorm(t *testing.T) {
	e := NewText(128)
	v := e.Embed("hello")
	if len(v) != 128 {
		t.Fatalf("dim = %d, want 128", len(v))
	}
	// Normalized vectors have unit length (a non-empty string has trigrams).
	var sumsq float32
	for _, x := range v {
		sumsq += x * x
	}
	if sumsq < 0.99 || sumsq > 1.01 {
		t.Fatalf("not normalized: sum of squares = %v", sumsq)
	}
}

// A typo must land closer to its source word than to an unrelated word. This is the
// property the demo relies on.
func TestSimilarStringsAreCloser(t *testing.T) {
	e := NewText(256)
	python := e.Embed("python")
	typo := e.Embed("pyton")
	unrelated := e.Embed("kubernetes")

	near, _ := vector.Cosine.Distance(typo, python)
	far, _ := vector.Cosine.Distance(typo, unrelated)
	if !(near < far) {
		t.Fatalf("typo not closer to source: d(pyton,python)=%v, d(pyton,kubernetes)=%v", near, far)
	}
}

func TestEmptyStringDoesNotPanic(t *testing.T) {
	e := NewText(64)
	v := e.Embed("")
	if len(v) != 64 {
		t.Fatalf("dim = %d, want 64", len(v))
	}
}
