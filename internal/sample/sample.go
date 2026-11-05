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

// Pick selects the next token. Reserved tokens (<pad>, <eos>, <unk>) are never chosen.
func Pick(logits []float32, o Options) int {
	best, bestV := -1, math.Inf(-1)
	for i := tokenizer.Unk + 1; i < len(logits); i++ {
		if v := float64(logits[i]); v > bestV {
			best, bestV = i, v
		}
	}
	if o.Temp <= 0 || o.Rng == nil {
		return best
	}
	idx := make([]int, 0, len(logits))
	for i := tokenizer.Unk + 1; i < len(logits); i++ {
		idx = append(idx, i)
	}
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	k := o.TopK
	if k <= 0 || k > len(idx) {
		k = len(idx)
	}
	idx = idx[:k]
	probs := make([]float64, k)
	var sum float64
	for i, id := range idx {
		probs[i] = math.Exp((float64(logits[id]) - bestV) / o.Temp)
		sum += probs[i]
	}
	r := o.Rng.Float64() * sum
	for i, p := range probs {
		r -= p
		if r <= 0 {
			return idx[i]
		}
	}
	return idx[k-1]
}
