// Package chat implements the multi-turn conversation loop on top of the model.
package chat

import (
	"strings"

	"github.com/rimsky-yamatov/gollm/internal/model"
	"github.com/rimsky-yamatov/gollm/internal/sample"
	"github.com/rimsky-yamatov/gollm/internal/tokenizer"
)
