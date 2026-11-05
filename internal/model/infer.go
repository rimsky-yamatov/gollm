package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/rimsky-yamatov/gollm/internal/tensor"
)

// ErrContextFull is returned by Step when the context window is exhausted.
var ErrContextFull = errors.New("gollm: context window is full")
