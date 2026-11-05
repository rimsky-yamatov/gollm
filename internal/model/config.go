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
