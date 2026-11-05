package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/rimsky-yamatov/gollm/internal/model"
	"github.com/rimsky-yamatov/gollm/internal/sample"
	"github.com/rimsky-yamatov/gollm/internal/tokenizer"
)

func TestBuildPromptDropsOldTurns(t *testing.T) {
	v := tokenizer.Build([]string{"ユーザー: ノックス\nあいうえお"})
	long := ""
	for i := 0; i < 80; i++ {
		long += "あ"
	}
	turns := []Turn{{long, long}, {"い", "う"}}
	p := BuildPrompt(turns, "え", v, 128)
	want := "ユーザー: い\nノックス: う\nユーザー: え\nノックス: "
	if p != want {
		t.Fatalf("prompt = %q", p)
	}
}

func TestBuildPromptNoHistory(t *testing.T) {
	v := tokenizer.Build([]string{"x"})
	p := BuildPrompt(nil, "こんにちは", v, 128)
	if p != "ユーザー: こんにちは\nノックス: " {
		t.Fatalf("prompt = %q", p)
	}
}

// TestGreedyMatchesReference decodes greedily and compares with the strings
// produced by tools/train_numpy.py golden.
func TestGreedyMatchesReference(t *testing.T) {
	root := filepath.Join("..", "..")
	b, err := os.ReadFile(filepath.Join(root, "testdata", "golden.json"))
	if err != nil {
		t.Skip("golden.json missing")
	}
	var g struct {
		Greedy [][]string `json:"greedy"`
	}
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	w, err := model.Load(filepath.Join(root, "model", "gollm.bin"))
	if err != nil {
		t.Skip("weights missing")
	}
	v, err := tokenizer.Load(filepath.Join(root, "model", "vocab.json"))
	if err != nil {
		t.Fatal(err)
	}
	sess := model.NewSession(w)
	for _, c := range g.Greedy {
		prompt := BuildPrompt(nil, c[1], v, w.Cfg.Ctx)
		got, err := Generate(sess, v, prompt, sample.Greedy(120))
		if err != nil {
			t.Fatal(err)
		}
		if got != c[2] {
			t.Errorf("%q:\n got  %q\n want %q", c[1], got, c[2])
		}
	}
}
