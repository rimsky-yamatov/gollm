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

// Turn is one user/assistant exchange.
type Turn struct{ User, Bot string }

// Chat keeps the dialogue history and generates replies.
type Chat struct {
	Sess  *model.Session
	Voc   *tokenizer.Vocab
	Opts  sample.Options
	turns []Turn
}

// New creates a Chat.
func New(w *model.Weights, v *tokenizer.Vocab, o sample.Options) *Chat {
	if o.MaxLen <= 0 {
		o.MaxLen = 120
	}
	return &Chat{Sess: model.NewSession(w), Voc: v, Opts: o}
}
