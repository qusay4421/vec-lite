// Command demo shows the whole stack end to end on real text: embed a small corpus,
// index it, and run similarity queries including misspellings. It uses the trigram
// text embedder, so this is lexical (fuzzy) matching; swapping in a real embedding
// model would make it semantic without changing the index or query code.
package main

import (
	"fmt"

	"github.com/qusay4421/vec-lite/internal/collection"
	"github.com/qusay4421/vec-lite/internal/embed"
	"github.com/qusay4421/vec-lite/internal/index"
	"github.com/qusay4421/vec-lite/internal/vector"
)

func main() {
	corpus := []struct{ term, kind string }{
		{"python", "language"}, {"javascript", "language"}, {"typescript", "language"},
		{"golang", "language"}, {"rust", "language"}, {"java", "language"},
		{"kubernetes", "infra"}, {"docker", "infra"}, {"terraform", "infra"},
		{"postgres", "database"}, {"redis", "database"}, {"mongodb", "database"},
		{"elasticsearch", "database"}, {"kafka", "infra"}, {"nginx", "infra"},
		{"tensorflow", "ml"}, {"pytorch", "ml"}, {"numpy", "ml"}, {"pandas", "ml"},
	}

	const dim = 256
	embedder := embed.NewText(dim)
	col := collection.New(dim, vector.Cosine, index.DefaultHNSWConfig())
	for _, c := range corpus {
		col.Add(c.term, embedder.Embed(c.term), collection.Metadata{"kind": c.kind})
	}
	fmt.Printf("indexed %d terms\n\n", col.Len())

	// Misspellings and partial strings: nearest neighbors should recover the intended
	// term by surface similarity.
	queries := []string{"pyton", "kubernates", "postgresql", "tensorflwo", "javascrpt"}
	for _, q := range queries {
		hits, _ := col.Search(embedder.Embed(q), 3)
		fmt.Printf("query %-12q -> ", q)
		for i, h := range hits {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%s (%.3f)", h.ID, h.Score)
		}
		fmt.Println()
	}

	// Filtered query: nearest database term to a misspelled "postgers".
	fmt.Println()
	hits, _ := col.SearchFiltered(embedder.Embed("postgers"), 3, collection.Equals("kind", "database"))
	fmt.Print("nearest databases to \"postgers\" -> ")
	for i, h := range hits {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%s", h.ID)
	}
	fmt.Println()
}
