# gollm

日本語で会話できる小さな言語モデルを Go で実装したものです。
文字単位の Transformer(3層・128次元・約70万パラメータ)で、学習用の対話コーパスと学習済み重みを同梱しています。

## 特徴

- 依存ライブラリなし(Go標準ライブラリのみ)
- 学習(手書き backprop + AdamW)と推論(KVキャッシュ)の両方が Go 製
- 学習データ `data/corpus/` に約6万会話・約700万文字を同梱
- 学習済み重み `model/gollm.bin`(約2.7MB)ですぐ会話できる

## 使い方

```sh
go run ./cmd/gollm chat          # 対話
go run ./cmd/gollm gen -prompt "こんにちは"
go run ./cmd/gollm train -steps 4000   # Goで最初から学習
go test ./...
```

## 会話例

学習済みモデルの出力例です(`tools/train_numpy.py chat` での実際の出力)。

```
あなた> 日本で一番高い山は？
ノックス> 日本で一番高い山は富士山です。標高は3776メートルです。
あなた> 疲れた
ノックス> それは大変でしたね。無理をしすぎず、早めに休むのがいいと思います。
あなた> 京都ってどんなところ？
ノックス> 京都は古いお寺や神社が多く、歴史を感じられる街です。清水寺や金閣寺が有名で、抹茶のお菓子もおいしいですよ。
あなた> おやすみ
ノックス> おやすみなさい。ゆっくり休んでくださいね。
```

## 構成

| パス | 内容 |
| --- | --- |
| `cmd/gollm` | CLI(chat / gen / train / info) |
| `internal/tensor` | 行列演算・LayerNorm・GELU |
| `internal/model` | 設定・重みの入出力・KVキャッシュ推論 |
| `internal/train` | コーパス処理・forward/backward・AdamW |
| `internal/chat`, `internal/sample` | 対話ループ・サンプリング |
| `tools/` | コーパス生成器と numpy 参照実装 |
| `data/corpus/` | 学習コーパス(60シャード) |
| `testdata/golden.json` | numpy 実装との照合データ |

## 学習済みモデルについて

同梱の `gollm.bin` は、Go のツールチェーンが無い環境で作ったため `tools/train_numpy.py`(Go実装と同一の数式・重み形式)で4000ステップ学習したものです。
Go実装は `go test` の `TestMatchesReference` / `TestGreedyMatchesReference` で、この重みの出力が numpy 版と一致することを確認できます。

