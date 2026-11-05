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
