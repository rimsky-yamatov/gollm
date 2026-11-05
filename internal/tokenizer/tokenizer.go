// Package tokenizer implements a character-level vocabulary.
// Every Unicode code point is one token; ids 0..2 are reserved.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
)
