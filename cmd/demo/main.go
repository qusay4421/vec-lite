// Command demo shows the whole stack end to end.
//
// With a pretrained word-vector file (-vectors, GloVe or word2vec text/.gz, free and
// offline) it does real semantic search: queries retrieve words by meaning. Without
// one it falls back to the trigram embedder, which matches by spelling (typo
// tolerance). Either way the index and query code are identical; only the embedder
// changes, which is the point.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/qusay4421/vec-lite/internal/collection"
	"github.com/qusay4421/vec-lite/internal/embed"
	"github.com/qusay4421/vec-lite/internal/index"
	"github.com/qusay4421/vec-lite/internal/vector"
)

func main() {
	vectorsPath := flag.String("vectors", os.Getenv("GLOVE_PATH"), "pretrained word vectors (GloVe/word2vec text or .gz) for semantic mode")
	flag.Parse()

	if *vectorsPath != "" {
		runSemantic(*vectorsPath)
		return
	}
	runLexical()
}

// runSemantic indexes single words and queries them by meaning, so a synonym that is
// not in the corpus still retrieves the related words.
func runSemantic(path string) {
	wv, err := embed.LoadWordVectorsFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load vectors: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("semantic mode: %d-dim vectors, vocab %d\n\n", wv.Dim(), wv.Vocab())

	corpus := []struct{ word, kind string }{
		{"dog", "animal"}, {"cat", "animal"}, {"puppy", "animal"}, {"kitten", "animal"},
		{"horse", "animal"}, {"elephant", "animal"},
		{"car", "vehicle"}, {"truck", "vehicle"}, {"bicycle", "vehicle"}, {"airplane", "vehicle"},
		{"king", "royalty"}, {"queen", "royalty"}, {"prince", "royalty"}, {"throne", "royalty"},
		{"happy", "emotion"}, {"sad", "emotion"}, {"angry", "emotion"},
		{"coffee", "food"}, {"tea", "food"}, {"bread", "food"}, {"pizza", "food"},
	}
	col := collection.New(wv.Dim(), vector.Cosine, index.DefaultHNSWConfig())
	for _, c := range corpus {
		col.Add(c.word, wv.Embed(c.word), collection.Metadata{"kind": c.kind})
	}

	// Query words are NOT in the corpus: retrieval is purely by learned meaning.
	queries := []string{"monarch", "automobile", "feline", "joyful", "espresso"}
	for _, q := range queries {
		hits, _ := col.Search(wv.Embed(q), 3)
		fmt.Printf("query %-11q -> ", q)
		printHits(hits)
	}

	fmt.Println()
	hits, _ := col.SearchFiltered(wv.Embed("beverage"), 3, collection.Equals("kind", "food"))
	fmt.Print("nearest food to \"beverage\" -> ")
	printHits(hits)
}

// runLexical is the no-dependency fallback: surface-form similarity via trigrams.
func runLexical() {
	fmt.Println("lexical mode (no -vectors given): matching by spelling, not meaning")
	fmt.Println("for semantic search, pass -vectors path/to/glove.txt (see scripts/get-glove.sh)")
	fmt.Println()

	corpus := []string{
		"python", "javascript", "typescript", "golang", "rust", "java",
		"kubernetes", "docker", "postgres", "redis", "tensorflow", "pytorch",
	}
	const dim = 256
	embedder := embed.NewText(dim)
	col := collection.New(dim, vector.Cosine, index.DefaultHNSWConfig())
	for _, term := range corpus {
		col.Add(term, embedder.Embed(term), nil)
	}
	for _, q := range []string{"pyton", "kubernates", "tensorflwo", "javascrpt"} {
		hits, _ := col.Search(embedder.Embed(q), 3)
		fmt.Printf("query %-12q -> ", q)
		printHits(hits)
	}
}

func printHits(hits []collection.Hit) {
	for i, h := range hits {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%s (%.3f)", h.ID, h.Score)
	}
	fmt.Println()
}
