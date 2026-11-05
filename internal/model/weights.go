package model

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
)

// Layer holds the parameters of one Transformer block.
// Matrices are [in*out] row-major so that y = x·W.
type Layer struct {
	Ln1G, Ln1B []float32
	Wq, Wk, Wv []float32
	Wo         []float32
	Ln2G, Ln2B []float32
	W1, B1     []float32
	W2, B2     []float32
}

// Weights is the complete parameter set. The output projection is tied to E.
type Weights struct {
	Cfg        Config
	E          []float32 // token embedding [Vocab*Dim]
	P          []float32 // position embedding [Ctx*Dim]
	L          []Layer
	LnfG, LnfB []float32
}

// Param is a named view onto one parameter tensor.
type Param struct {
	Name  string
	Data  []float32
	Decay bool // apply weight decay (2-D weight matrices except embeddings)
}
