// Package chat implements the multi-turn conversation loop on top of the model.
package chat

import (
	"strings"

	"github.com/rimsky-yamatov/gollm/internal/model"
	"github.com/rimsky-yamatov/gollm/internal/sample"
	"github.com/rimsky-yamatov/gollm/internal/tokenizer"
)

const (
	userPrefix = "ユーザー: "
	botTag     = "ノックス: "
	// replyReserve is the number of context tokens kept free for the reply.
	replyReserve = 70
)
