// Package embed turns short text into vectors using hashed character trigrams.
//
// This is lexical (surface-form) similarity, not learned semantics: it places
// strings that share character sequences near each other, which is enough to
// demo retrieval (fuzzy matching, typo tolerance) with no ML model or network
// dependency. To get true semantic search, replace this embedder with a real
// model; nothing in the index or collection changes, because both only ever see
// vectors.
package embed

import (
	"hash/fnv"
	"strings"

	"github.com/qusay4421/vec-lite/internal/vector"
)

// Text embeds strings into a fixed-dimension vector via the hashing trick: each
// character trigram is hashed to a dimension and counted. Fixed width means no
// vocabulary to manage and every string maps to the same dimensionality.
type Text struct {
	dim int
}

func NewText(dim int) *Text {
	if dim < 1 {
		dim = 256
	}
	return &Text{dim: dim}
}

func (t *Text) Dim() int { return t.dim }

// Embed returns the L2-normalized trigram histogram of s. Normalizing makes cosine
// distance compare shape rather than string length, so "cat" and "cats" stay close.
func (t *Text) Embed(s string) vector.Vector {
	v := make(vector.Vector, t.dim)
	// Pad with spaces so very short strings still yield trigrams and word boundaries
	// contribute, which helps prefix/suffix similarity.
	runes := []rune("  " + strings.ToLower(s) + "  ")
	for i := 0; i+3 <= len(runes); i++ {
		h := hashTrigram(runes[i : i+3])
		v[h%uint32(t.dim)]++
	}
	return vector.Normalize(v)
}

func hashTrigram(tri []rune) uint32 {
	h := fnv.New32a()
	var buf [4]byte
	for _, r := range tri {
		buf[0] = byte(r)
		buf[1] = byte(r >> 8)
		buf[2] = byte(r >> 16)
		buf[3] = byte(r >> 24)
		h.Write(buf[:])
	}
	return h.Sum32()
}
