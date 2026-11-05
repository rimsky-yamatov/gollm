// Package sample turns logits into token ids.
package sample

import (
	"math"
	"math/rand"
	"sort"

	"github.com/rimsky-yamatov/gollm/internal/tokenizer"
)

// Options controls decoding. Temp <= 0 selects greedy decoding.
type Options struct {
	Temp   float64
	TopK   int
	MaxLen int
	Rng    *rand.Rand
}

// Greedy returns deterministic decoding options.
func Greedy(maxLen int) Options { return Options{Temp: 0, TopK: 1, MaxLen: maxLen} }
