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

// TestMatchesReference checks that the Go forward pass reproduces the logits
// computed by the numpy reference implementation (tools/train_numpy.py).
func TestMatchesReference(t *testing.T) {
	g, w, v := loadGolden(t)
	ids := v.Encode(g.Prompt)
	if len(ids) != len(g.IDs) {
		t.Fatalf("tokenisation length %d != %d", len(ids), len(g.IDs))
	}
	for i := range ids {
		if ids[i] != g.IDs[i] {
			t.Fatalf("token %d: %d != %d", i, ids[i], g.IDs[i])
		}
	}
	s := NewSession(w)
	logits, err := s.Feed(ids)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range g.LastLogitsHead {
		if d := math.Abs(float64(logits[i]) - want); d > 5e-3 {
			t.Errorf("logit[%d] = %v, want %v (diff %g)", i, logits[i], want, d)
		}
	}
	best := 0
	for i, x := range logits {
		if x > logits[best] {
			best = i
		}
	}
	if best != g.LastArgmax {
		t.Errorf("argmax = %d, want %d", best, g.LastArgmax)
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	cfg := Config{Vocab: 9, Dim: 8, Layers: 2, Heads: 2, FF: 16, Ctx: 5}
	w := New(cfg, newRand(3))
	path := filepath.Join(t.TempDir(), "w.bin")
	if err := Save(path, w); err != nil {
		t.Fatal(err)
	}
	w2, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if w2.Cfg != cfg {
		t.Fatalf("cfg = %+v", w2.Cfg)
	}
	a, b := w.Params(), w2.Params()
	for i := range a {
		for j := range a[i].Data {
			if a[i].Data[j] != b[i].Data[j] {
				t.Fatalf("%s[%d] differs", a[i].Name, j)
			}
		}
	}
}

func TestNumParamsMatchesTensors(t *testing.T) {
	cfg := Default(670)
	w := Zeros(cfg)
	n := 0
	for _, p := range w.Params() {
		n += len(p.Data)
	}
	if n != cfg.NumParams() {
		t.Fatalf("tensors hold %d params, NumParams says %d", n, cfg.NumParams())
	}
}
