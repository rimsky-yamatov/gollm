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

// Reset forgets the conversation.
func (c *Chat) Reset() { c.turns = nil }

// BuildPrompt renders the history plus the new user line. The oldest turns are
// dropped until the prompt leaves room for a reply inside the context window.
func BuildPrompt(turns []Turn, user string, v *tokenizer.Vocab, ctx int) string {
	budget := ctx - replyReserve
	for start := 0; ; start++ {
		var b strings.Builder
		for _, t := range turns[start:] {
			b.WriteString(userPrefix + t.User + "\n" + botTag + t.Bot + "\n")
		}
		b.WriteString(userPrefix + user + "\n" + botTag)
		p := b.String()
		if start >= len(turns) || len(v.Encode(p)) <= budget {
			return p
		}
	}
}

func (c *Chat) prompt(user string) string {
	return BuildPrompt(c.turns, user, c.Voc, c.Sess.W.Cfg.Ctx)
}

// Reply generates the assistant's answer to user and records the turn.
func (c *Chat) Reply(user string) (string, error) {
	user = strings.TrimSpace(strings.ReplaceAll(user, "\n", " "))
	out, err := Generate(c.Sess, c.Voc, c.prompt(user), c.Opts)
	if err != nil {
		return "", err
	}
	c.turns = append(c.turns, Turn{user, out})
	return out, nil
}

// Generate feeds prompt into a fresh session and decodes until a newline.
func Generate(s *model.Session, v *tokenizer.Vocab, prompt string, o sample.Options) (string, error) {
	ids := v.Encode(prompt)
	ctx := s.W.Cfg.Ctx
	if len(ids) > ctx-8 {
		ids = ids[len(ids)-(ctx-8):]
	}
	s.Reset()
	logits, err := s.Feed(ids)
	if err != nil {
		return "", err
	}
	var out []string
	for i := 0; i < o.MaxLen; i++ {
		id := sample.Pick(logits, o)
		tok := v.Token(id)
		if tok == "\n" {
			break
		}
		out = append(out, tok)
		if s.Pos() >= ctx {
			break
		}
		if logits, err = s.Step(id); err != nil {
			return "", err
		}
	}
	return strings.Join(out, ""), nil
}
