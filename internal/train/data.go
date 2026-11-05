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
