// Package sample turns logits into token ids.
package sample

import (
	"math"
	"math/rand"
	"sort"

	"github.com/rimsky-yamatov/gollm/internal/tokenizer"
)
