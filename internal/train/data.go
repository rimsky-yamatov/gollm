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
