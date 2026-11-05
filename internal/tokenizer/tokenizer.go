// Package tokenizer implements a character-level vocabulary.
// Every Unicode code point is one token; ids 0..2 are reserved.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

// Reserved token ids.
const (
	Pad = 0
	EOS = 1
	Unk = 2
)

// Vocab maps between code points (as strings) and ids.
type Vocab struct {
	Tokens []string
	index  map[string]int
}

// New builds a Vocab from an explicit token list.
func New(tokens []string) *Vocab {
	v := &Vocab{Tokens: tokens, index: make(map[string]int, len(tokens))}
	for i, t := range tokens {
		v.index[t] = i
	}
	return v
}

// Build creates a vocabulary from texts: the reserved tokens followed by every
// distinct rune in ascending code point order.
func Build(texts []string) *Vocab {
	seen := map[rune]bool{}
	for _, t := range texts {
		for _, r := range t {
			seen[r] = true
		}
	}
	rs := make([]rune, 0, len(seen))
	for r := range seen {
		rs = append(rs, r)
	}
	sort.Slice(rs, func(i, j int) bool { return rs[i] < rs[j] })
	toks := []string{"<pad>", "<eos>", "<unk>"}
	for _, r := range rs {
		toks = append(toks, string(r))
	}
	return New(toks)
}

// Size returns the number of tokens.
func (v *Vocab) Size() int { return len(v.Tokens) }

// Encode converts text to ids; unknown characters map to Unk.
func (v *Vocab) Encode(s string) []int {
	out := make([]int, 0, len(s)/2)
	for _, r := range s {
		if id, ok := v.index[string(r)]; ok {
			out = append(out, id)
		} else {
			out = append(out, Unk)
		}
	}
	return out
}
