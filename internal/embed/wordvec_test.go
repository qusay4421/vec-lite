package embed

import (
	"strings"
	"testing"

	"github.com/qusay4421/vec-lite/internal/vector"
)

// A small synthetic table keeps the test offline and deterministic while still
// exercising load, header detection, mean-pooling, and semantic ordering.
const gloveStyle = `king 1.0 0.0
queen 0.9 0.1
apple 0.0 1.0
the 0.0 0.0`

const word2vecStyle = `3 2
king 1.0 0.0
queen 0.9 0.1
apple 0.0 1.0`

func TestLoadGloVeFormat(t *testing.T) {
	wv, err := LoadWordVectors(strings.NewReader(gloveStyle))
	if err != nil {
		t.Fatal(err)
	}
	if wv.Dim() != 2 || wv.Vocab() != 4 {
		t.Fatalf("dim=%d vocab=%d, want 2 and 4", wv.Dim(), wv.Vocab())
	}
}

// The word2vec header line must be detected and skipped, not parsed as a word.
func TestLoadWord2VecHeaderSkipped(t *testing.T) {
	wv, err := LoadWordVectors(strings.NewReader(word2vecStyle))
	if err != nil {
		t.Fatal(err)
	}
	if wv.Vocab() != 3 {
		t.Fatalf("vocab=%d, want 3 (header should be skipped)", wv.Vocab())
	}
	if _, leaked := wv.vecs["3"]; leaked {
		t.Fatal("header line was parsed as a word")
	}
}

// Embedding averages the known words, and ignores unknown ones.
func TestEmbedMeanPools(t *testing.T) {
	wv, _ := LoadWordVectors(strings.NewReader(gloveStyle))
	// "king zzz": zzz is unknown and skipped, so the mean of {king} = king.
	v := wv.Embed("king zzz")
	if v[0] < 0.99 || v[1] > 0.01 {
		t.Fatalf("embed = %v, want ~king (1,0)", v)
	}
	// A known zero-vector word like "the" still counts, pulling the mean toward
	// origin: mean of king(1,0) and the(0,0) is (0.5,0).
	half := wv.Embed("king the")
	if half[0] < 0.49 || half[0] > 0.51 {
		t.Fatalf("embed(king the) = %v, want ~(0.5,0)", half)
	}
	// No known words -> zero vector.
	if z := wv.Embed("zzz qqq"); z[0] != 0 || z[1] != 0 {
		t.Fatalf("unknown-only embed = %v, want zero", z)
	}
}

// Meaning, not spelling: "king" must sit closer to "queen" than to "apple", which is
// what makes this real semantic retrieval given proper vectors.
func TestSemanticOrdering(t *testing.T) {
	wv, _ := LoadWordVectors(strings.NewReader(gloveStyle))
	king := wv.Embed("king")
	near, _ := vector.Cosine.Distance(king, wv.Embed("queen"))
	far, _ := vector.Cosine.Distance(king, wv.Embed("apple"))
	if !(near < far) {
		t.Fatalf("king not closer to queen than apple: d(queen)=%v d(apple)=%v", near, far)
	}
}
