// Package train contains the corpus pipeline and the from-scratch
// forward/backward implementation used to train gollm in pure Go.
package train

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/rimsky-yamatov/gollm/internal/tokenizer"
)

// BotPrefix is the speaker tag of the assistant; tokens of lines starting
// with it are the only ones that contribute to the loss.
const (
	BotPrefix  = "ノックス:"
	UserPrefix = "ユーザー: "
	BotTag     = "ノックス: "
)

// LoadConversations reads every *.txt file in dir. Conversations are separated
// by blank lines and each one is normalised to end with exactly one "\n".
func LoadConversations(dir string) ([]string, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.txt"))
	if err != nil {
		return nil, err
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no *.txt files in %s", dir)
	}
	sort.Strings(files)
	var convs []string
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		for _, c := range strings.Split(string(b), "\n\n") {
			c = strings.Trim(c, "\n")
			if c != "" {
				convs = append(convs, c+"\n")
			}
		}
	}
	return convs, nil
}

// Stream is the whole corpus as one token sequence with a loss mask.
type Stream struct {
	IDs    []int32
	Mask   []float32
	Starts []int // token offset of each conversation
}
