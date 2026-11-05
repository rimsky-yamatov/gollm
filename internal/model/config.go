// Package model defines the gollm Transformer: configuration, weights,
// serialisation and cached incremental inference.
package model

import "fmt"

// Config describes the architecture. The defaults match model/gollm.bin.
type Config struct {
	Vocab  int // number of tokens
	Dim    int // model width
	Layers int // number of Transformer blocks
	Heads  int // attention heads (Dim must be divisible by Heads)
	FF     int // hidden width of the MLP
	Ctx    int // maximum context length (learned positions)
}

// Default returns the architecture of the bundled model for a vocab size.
func Default(vocab int) Config {
	return Config{Vocab: vocab, Dim: 128, Layers: 3, Heads: 4, FF: 512, Ctx: 128}
}

// HeadDim is the per-head width.
func (c Config) HeadDim() int { return c.Dim / c.Heads }
