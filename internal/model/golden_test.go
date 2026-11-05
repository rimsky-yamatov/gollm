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
