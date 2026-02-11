# テキストエンコーダだけを別モデルに差し替える方法

diffusers のパイプラインはコンポーネント（transformer, text_encoder, tokenizer, vae, scheduler 等）を組み合わせた構成なので、テキストエンコーダだけを差し替える方法は主に **3パターン** あります。

---

## パターン1: `from_pretrained` 時にコンポーネントを渡す

パイプラインのロード時に、別途ロードしたテキストエンコーダとトークナイザを明示的に渡す方法です。最もクリーンなやり方です。

```python
from transformers import AutoTokenizer, AutoModel  # or CLIPTextModel, T5EncoderModel etc.

# 差し替えたいテキストエンコーダを個別にロード
custom_tokenizer = AutoTokenizer.from_pretrained("your/custom-text-encoder")
custom_text_encoder = AutoModel.from_pretrained(
    "your/custom-text-encoder", torch_dtype=dtype
)

# パイプラインに渡す（元のテキストエンコーダの代わりに使われる）
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=transformer,
    scheduler=scheduler,
    text_encoder=custom_text_encoder,    # ← 差し替え
    tokenizer=custom_tokenizer,          # ← 対応するトークナイザも
    torch_dtype=dtype,
)
```

FLUX.2 Klein の場合、テキストエンコーダが複数ある（`text_encoder`, `text_encoder_2` 等）パイプラインでは、差し替えたい方だけ指定します。

```python
pipeline = Flux2KleinPipeline.from_pretrained(
    MODEL_ID,
    text_encoder=custom_text_encoder,      # 1つ目だけ差し替え
    tokenizer=custom_tokenizer,
    # text_encoder_2 は元のまま
    torch_dtype=dtype,
)
```

---

## パターン2: ロード後にアトリビュートを直接差し替え

パイプラインをロードした後で、属性を上書きする方法です。手軽ですが、デバイスやdtype の整合性は自分で管理する必要があります。

```python
# 通常通りパイプラインをロード
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    BASE_MODEL_ID, transformer=transformer, torch_dtype=dtype
)

# 後から差し替え
pipeline.tokenizer = AutoTokenizer.from_pretrained("your/custom-text-encoder")
pipeline.text_encoder = AutoModel.from_pretrained(
    "your/custom-text-encoder", torch_dtype=dtype
).to(pipeline.device)
```

---

## パターン3: サブコンポーネントだけローカル保存してパスで指定

diffusers のパイプラインは各サブコンポーネントをサブフォルダで管理しているので、ローカルにディレクトリ構造を作って差し替える方法もあります。

```
my_custom_pipeline/
├── model_index.json          ← 元モデルからコピー
├── text_encoder/             ← 差し替えたいモデルの重みを配置
├── tokenizer/                ← 差し替えたいトークナイザを配置
├── transformer/              → 元モデルへのシンボリックリンク or コピー
├── vae/                      → 元モデルへのシンボリックリンク or コピー
└── scheduler/                → 元モデルへのシンボリックリンク or コピー
```

```python
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "./my_custom_pipeline", torch_dtype=dtype
)
```

---

## 注意点

- **hidden_size の互換性**: transformer が期待するテキスト埋め込みの次元（hidden_size）と、差し替えるテキストエンコーダの出力次元が一致している必要があります。不一致の場合、線形射影レイヤーを挟む必要があります
- **トークナイザの整合性**: テキストエンコーダとトークナイザは必ずセットで差し替えてください。片方だけ変えると vocab の不一致でエラーになります
- **max_length**: パイプライン内部でトークンの最大長をハードコードしている場合があるので、差し替え先のモデルに合わせて調整が必要な場合があります
- **offload との相性**: `enable_model_cpu_offload()` や `enable_sequential_cpu_offload()` はパイプラインの全コンポーネントを対象にするので、差し替え後に呼べば問題ありません。差し替え前に呼んでいた場合は、offload の再設定が必要です

最も現実的なのは **パターン1** で、`from_pretrained` のキーワード引数として渡す方法です。
