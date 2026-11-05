package tokenizer

import "testing"

func TestRoundTrip(t *testing.T) {
	v := Build([]string{"こんにちは\n", "ユーザー: 元気？"})
	s := "こんにちは、元気？\n"
	ids := v.Encode(s)
	if len(ids) != len([]rune(s)) {
		t.Fatalf("len = %d", len(ids))
	}
	got := v.Decode(ids)
	// "、" is not in the vocabulary, so it must be dropped as <unk>.
	if got != "こんにちは元気？\n" {
		t.Fatalf("decode = %q", got)
	}
}

func TestReservedIDs(t *testing.T) {
	v := Build([]string{"あ"})
	if v.Token(Pad) != "<pad>" || v.Token(EOS) != "<eos>" || v.Token(Unk) != "<unk>" {
		t.Fatal("reserved ids wrong")
	}
	if v.Encode("あ")[0] != 3 {
		t.Fatal("first real token must have id 3")
	}
}

func TestSortedByCodePoint(t *testing.T) {
	v := Build([]string{"ba"})
	if v.Token(3) != "a" || v.Token(4) != "b" {
		t.Fatalf("tokens = %v", v.Tokens)
	}
}
