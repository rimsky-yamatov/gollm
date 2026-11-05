package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rimsky-yamatov/gollm/internal/tokenizer"
)

type golden struct {
	Prompt         string      `json:"prompt"`
	IDs            []int       `json:"ids"`
	LastLogitsHead []float64   `json:"last_logits_head"`
	LastArgmax     int         `json:"last_argmax"`
	Greedy         [][]string  `json:"greedy"`
}

func root() string { return filepath.Join("..", "..") }

func loadGolden(t *testing.T) (*golden, *Weights, *tokenizer.Vocab) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(root(), "testdata", "golden.json"))
	if err != nil {
		t.Skipf("golden.json missing: %v", err)
	}
	var g golden
	if err := json.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	w, err := Load(filepath.Join(root(), "model", "gollm.bin"))
	if err != nil {
		t.Skipf("weights missing: %v", err)
	}
	v, err := tokenizer.Load(filepath.Join(root(), "model", "vocab.json"))
	if err != nil {
		t.Fatal(err)
	}
	return &g, w, v
}
