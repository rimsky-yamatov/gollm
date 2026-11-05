package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/rimsky-yamatov/gollm/internal/model"
	"github.com/rimsky-yamatov/gollm/internal/sample"
	"github.com/rimsky-yamatov/gollm/internal/tokenizer"
)

func TestBuildPromptDropsOldTurns(t *testing.T) {
	v := tokenizer.Build([]string{"ユーザー: ノックス\nあいうえお"})
	long := ""
	for i := 0; i < 80; i++ {
		long += "あ"
	}
	turns := []Turn{{long, long}, {"い", "う"}}
	p := BuildPrompt(turns, "え", v, 128)
	want := "ユーザー: い\nノックス: う\nユーザー: え\nノックス: "
	if p != want {
		t.Fatalf("prompt = %q", p)
	}
}

func TestBuildPromptNoHistory(t *testing.T) {
	v := tokenizer.Build([]string{"x"})
	p := BuildPrompt(nil, "こんにちは", v, 128)
	if p != "ユーザー: こんにちは\nノックス: " {
		t.Fatalf("prompt = %q", p)
	}
}
