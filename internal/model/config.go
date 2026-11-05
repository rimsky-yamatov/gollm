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

// Validate checks the configuration for consistency.
func (c Config) Validate() error {
	switch {
	case c.Vocab < 4:
		return fmt.Errorf("vocab too small: %d", c.Vocab)
	case c.Dim <= 0 || c.Layers <= 0 || c.Heads <= 0 || c.FF <= 0 || c.Ctx <= 0:
		return fmt.Errorf("invalid config: %+v", c)
	case c.Dim%c.Heads != 0:
		return fmt.Errorf("dim %d not divisible by heads %d", c.Dim, c.Heads)
	}
	return nil
}
