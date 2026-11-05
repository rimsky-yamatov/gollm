// Command gollm is a small Japanese chat language model written in Go.
//
//	gollm chat   [-model model/gollm.bin] [-vocab model/vocab.json]
//	gollm gen    -prompt "こんにちは"
//	gollm train  [-data data/corpus] [-steps 4000] [-resume]
//	gollm info
package main

import (
	"bufio"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/rimsky-yamatov/gollm/internal/chat"
	"github.com/rimsky-yamatov/gollm/internal/model"
	"github.com/rimsky-yamatov/gollm/internal/sample"
	"github.com/rimsky-yamatov/gollm/internal/tokenizer"
	"github.com/rimsky-yamatov/gollm/internal/train"
)
