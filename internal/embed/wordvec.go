package embed

import (
	"bufio"
	"compress/gzip"
	"io"
	"os"
	"strconv"
	"strings"
	"unicode"

	"github.com/qusay4421/vec-lite/internal/vector"
)

// WordVectors embeds text with pretrained word vectors (GloVe or word2vec text
// format), averaging the vectors of the words it contains.
//
// Unlike the trigram embedder, this captures meaning, not spelling: "monarch" lands
// near "king" because their pretrained vectors are close, even though they share no
// characters. Mean-pooling word vectors is a classic, dependency-free way to embed a
// short text, and the vector files are free and run fully offline (no API, no key).
// It is weaker than a sentence-transformer on word order and negation, which is the
// honest tradeoff for zero cost.
type WordVectors struct {
	dim  int
	vecs map[string]vector.Vector
}

// LoadWordVectors reads vectors in GloVe or word2vec text format. Each line is
// "word f1 f2 ... fd". A word2vec file starts with a "<count> <dim>" header line,
// which is detected and skipped; GloVe files have no header.
func LoadWordVectors(r io.Reader) (*WordVectors, error) {
	wv := &WordVectors{vecs: make(map[string]vector.Vector)}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024) // long lines for high-dim vectors

	first := true
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) == 0 {
			continue
		}
		// A two-field first line of integers is a word2vec header (count, dim).
		if first {
			first = false
			if len(fields) == 2 {
				if _, e1 := strconv.Atoi(fields[0]); e1 == nil {
					if _, e2 := strconv.Atoi(fields[1]); e2 == nil {
						continue
					}
				}
			}
		}
		word := fields[0]
		vals := fields[1:]
		if wv.dim == 0 {
			wv.dim = len(vals)
		}
		if len(vals) != wv.dim {
			continue // skip a malformed line rather than corrupt the table
		}
		vec := make(vector.Vector, wv.dim)
		for i, s := range vals {
			f, err := strconv.ParseFloat(s, 32)
			if err != nil {
				vec = nil
				break
			}
			vec[i] = float32(f)
		}
		if vec != nil {
			wv.vecs[word] = vec
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return wv, nil
}

// LoadWordVectorsFile loads from a path, transparently decompressing a .gz file so
// the common gzipped GloVe download works without a manual unzip step.
func LoadWordVectorsFile(path string) (*WordVectors, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var r io.Reader = f
	if strings.HasSuffix(path, ".gz") {
		gz, err := gzip.NewReader(f)
		if err != nil {
			return nil, err
		}
		defer gz.Close()
		r = gz
	}
	return LoadWordVectors(r)
}

func (wv *WordVectors) Dim() int   { return wv.dim }
func (wv *WordVectors) Vocab() int { return len(wv.vecs) }

// Embed returns the mean of the vectors of the known words in text. Unknown words
// are skipped; a text with no known words returns the zero vector (maximally far
// under cosine), which is the right "no signal" answer.
func (wv *WordVectors) Embed(text string) vector.Vector {
	out := make(vector.Vector, wv.dim)
	found := 0
	for _, tok := range tokenize(text) {
		if v, ok := wv.vecs[tok]; ok {
			for i := range out {
				out[i] += v[i]
			}
			found++
		}
	}
	if found == 0 {
		return out
	}
	inv := 1 / float32(found)
	for i := range out {
		out[i] *= inv
	}
	return out
}

// tokenize lowercases and splits on any non-alphanumeric rune, matching how GloVe
// vocabularies are cased and punctuated.
func tokenize(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
}
