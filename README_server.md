# Web Server ユーザーガイド

AI画像編集Webサーバーの機能説明と操作方法です。

## サーバーの種類

| サーバー | スクリプト | モデル | 特徴 |
|---------|-----------|--------|------|
| **Nunchaku** | `app_nunchaku.py` | Qwen-Image-Edit-2509 Lightning | SVDQ量子化で高速。RTX 3xxx (12GB) で動作 |
| **GGUF** | `app_gguf.py` | Qwen-Image-Edit-Rapid-AIO-V23 (GGUF) | Q3_K～Q8_0の量子化レベルを選択可能。低VRAM向き |
| **AIO** | `app_aio.py` | Qwen-Image-Edit-Rapid-AIO-V23 | bf16フル精度。VRAM多め必要 |
| **ComfyUI (AIO)** | `app_comfyui.py` | ComfyUI API経由 (AIOワークフロー) | torch/diffusers不要。別途ComfyUI実行が必要 |
| **ComfyUI (Nunchaku)** | `app_comfyui_nunchaku.py` | ComfyUI API経由 (Nunchakuワークフロー) | torch/diffusers不要。別途ComfyUI実行が必要 |
| **ComfyUI (GGUF)** | `app_comfyui_gguf.py` | ComfyUI API経由 (GGUFワークフロー) | torch/diffusers不要。GGUF量子化モデル使用 |

## 起動方法

```powershell
cd server

# 基本起動（パスワード: password, ポート: 5000）
python app_nunchaku.py

# パスワード・ポート指定
python app_nunchaku.py --password mysecret --port 8080

# ギャラリーモード有効化（履歴・共有機能）
python app_nunchaku.py --gallery --password mysecret

# LoRA指定（複数可）
python app_nunchaku.py --lora "path/to/lora.safetensors"
python app_nunchaku.py --lora "repo_id::weight.safetensors" --lora "another.safetensors"

# プロンプトプリセット
python app_nunchaku.py --preset "高画質化::Enhance quality." --preset "テキスト除去::Remove all text."

# 外部公開（LAN内の他端末からアクセス）
python app_nunchaku.py --host 0.0.0.0
```

### ComfyUI版の起動

ComfyUI版はtorch/diffusersに依存せず、別途起動中のComfyUIインスタンスにAPI経由で推論を委譲します。

```powershell
cd server

# ComfyUI (AIOワークフロー)
python app_comfyui.py
python app_comfyui.py --password mysecret --port 5000
python app_comfyui.py --comfyui-url http://192.168.1.100:8188
python app_comfyui.py --comfyui-path D:\ComfyUI    # LoRAパス自動登録 + ComfyUI再起動
python app_comfyui.py --steps 4 --cfg 1.0

# ComfyUI (Nunchakuワークフロー)
python app_comfyui_nunchaku.py
python app_comfyui_nunchaku.py --password mysecret --port 5000
python app_comfyui_nunchaku.py --comfyui-url http://192.168.1.100:8188
python app_comfyui_nunchaku.py --comfyui-path D:\ComfyUI

# ComfyUI (GGUFワークフロー)
python app_comfyui_gguf.py
python app_comfyui_gguf.py --password mysecret --port 5000
python app_comfyui_gguf.py --comfyui-url http://192.168.1.100:8188
python app_comfyui_gguf.py --comfyui-path D:\ComfyUI
```

### 起動オプション一覧

#### 共通オプション
| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--host HOST` | バインドホスト | `127.0.0.1` |
| `--port PORT` | バインドポート | `5000` |
| `--password PW` | 生成用パスワード | `password` |
| `--no-progress` | HFダウンロード進捗を非表示 | 表示 |
| `--no-offload` | オフロード無効（高VRAM向け） | 自動 |
| `--lora REPO_OR_PATH` | LoRA重み（複数指定可）。`パス` or `repo_id::weight_name` | なし |
| `--gallery` | ギャラリーモード有効化 | 無効 |
| `--preset "label::prompt"` | プリセットボタン（複数指定可） | なし |

#### サーバー固有オプション
| オプション | 対象 | 説明 |
|-----------|------|------|
| `--steps N` | Nunchaku | 推論ステップ数: 4 or 8（デフォルト: 8） |
| `--rank N` | Nunchaku | Nunchakuランク: 32 or 128 |
| `--num-blocks-on-gpu N` | Nunchaku | 低VRAMモードでGPU上に保持するブロック数 |
| `--gguf-file PATH` | GGUF | HFリポジトリ内のGGUFファイル |
| `--gguf-local PATH` | GGUF | ローカルGGUFファイルを直接使用 |
| `--offload` | AIO | 逐次CPUオフロード（低VRAM向け） |
| `--comfyui-url URL` | ComfyUI両版 | ComfyUI API URL（デフォルト: `http://127.0.0.1:8188`） |
| `--comfyui-path DIR` | ComfyUI両版 | ComfyUIインストールディレクトリ。LoRAパスを自動登録してComfyUIを再起動 |
| `--steps N` | ComfyUI両版 | 推論ステップ数（デフォルト: 8） |
| `--cfg N` | ComfyUI両版 | CFGスケール（デフォルト: 1.0） |

> **注意:** GGUFサーバーは `--offload` 非対応（GGUF tensorsと互換性なし）。
> **注意:** AIOサーバーはGGUFより大幅にVRAMが必要（bf16フル精度）。
> **注意:** ComfyUI版は別途ComfyUIの起動が必要です。起動時に接続を確認し、失敗すると60秒後にリトライします。
> **注意:** ComfyUI (Nunchaku) 版はNunchakuライブラリとカスタムノード `nunchaku-ai/ComfyUI-nunchaku`、`ussoewwin/ComfyUI-QwenImageLoraLoader` のインストールが必要です。
> **注意:** ComfyUI (GGUF) 版はggufライブラリとカスタムノード `city96/ComfyUI-GGUF` のインストールが必要です。

---

## 基本操作

### 1. ログイン

ギャラリーモード（`--gallery`）の場合、最初にパスワード入力画面が表示されます。起動時に指定したパスワードを入力してログインしてください。

ギャラリーモードなしの場合はログイン画面なしで直接フォームが表示されます（パスワードはフォーム内で入力）。

### 2. 画像編集（img2img）

1. **Password** 欄にパスワードを入力
2. **Image 1** に編集したい画像をアップロード（ファイル選択 or ドラッグ&ドロップ）
3. **Image 2 / REF** に参考画像を追加（任意）
4. **Prompt** に編集指示を入力（空欄の場合はデフォルトプロンプト使用）
5. **Pre-resize** で入力画像のサイズを選択。Nunchaku/GGUF/AIO版は 0.3M / 1M pixels、ComfyUI版は 1M / 2M pixels
6. **Generate** ボタンをクリック

### 3. テキストから画像生成（t2i）

1. **Text-to-Image (t2i) mode** チェックボックスをON
2. 画像入力エリアが非表示になる
3. **Prompt** に生成したい画像の説明を入力（必須）
4. **Generate** をクリック → 1024x1024の画像が生成される

### 4. 生成の進捗とキャンセル

- 生成中はステップ進捗がリアルタイム表示される（例: `Processing... (step 3/8)`）
- プログレスバーで視覚的に進捗を確認可能
- **Cancel** ボタンで生成を中断できる
- ジョブキュー: 同時に1件処理 + 2件待機。3件以上の同時投入は `503 BUSY` エラー
- **プレビュー表示**（ComfyUI版のみ）: 「Show preview during generation」チェックボックスで、生成中のプレビュー画像をリアルタイム表示。ComfyUIの Settings → Preview Method を有効にする必要あり

---

## プロンプト関連機能

### プリセットボタン

`--preset` オプションで設定したプリセットがプロンプト欄上部にボタン表示されます。

- クリックでプリセットのプロンプトテキストが入力される
- プロンプト欄に既にテキストがある場合は確認ダイアログが表示される
- 例: `--preset "高画質化::Enhance quality."` → 「高画質化」ボタン

### プロンプトクリア

プロンプトラベル横の **×** ボタンでプロンプトを全消去できます。

### プロンプト翻訳

プロンプト欄下部の翻訳ボタンで、入力中のプロンプトを翻訳できます。

| ボタン | 機能 |
|--------|------|
| `→ EN` | 英語に翻訳 |
| `→ ZH` | 中国語に翻訳 |
| `→ JA` | 日本語に翻訳 |

モデルによっては英語や中国語のプロンプトのほうが良い結果が出ることがあるため、翻訳機能が用意されています。

---

## 連続修正モード (Continuous Mode)

**Continuous mode** チェックボックスをONにすると、画像生成完了時に結果画像が自動的にImg1に設定されます。

これにより、生成結果に対してさらにプロンプトを変えて繰り返し編集する「連続修正」ワークフローが簡単になります。

また、生成結果エリアには **Img1** / **Img2** ラジオボタンが表示され、ギャラリーの更新を待たずに即座に生成結果を次の入力として使用できます。

---

## LoRA

### 設定方法

**Nunchaku / GGUF / AIO版:** 起動時に `--lora` オプションで指定します（複数指定可）。

```powershell
# ローカルファイル
python app_nunchaku.py --lora "path/to/lora.safetensors"

# HuggingFace リポジトリ
python app_nunchaku.py --lora "repo_id::weight_name.safetensors"

# 複数LoRA
python app_nunchaku.py --lora "lora1.safetensors" --lora "lora2.safetensors"
```

**ComfyUI版:** `server/LoRA/` フォルダにLoRAファイルを配置します。`--comfyui-path` を指定すると `extra_model_paths.yaml` にパスが自動登録されます。コマンドラインでの `--lora` 指定は不要です。

### WebUIでの操作

- 登録されたLoRAはフォームにチェックボックス + 強度スライダー（0.0〜2.0）として表示
- デフォルトはすべてチェックOFF
- 生成ごとにチェックの ON/OFF と強度を変更可能

### サーバー別の動作

| サーバー | LoRA切り替え方式 | 速度 |
|---------|-----------------|------|
| **Nunchaku** | パイプライン全体リロード（構成変更時のみ） | 遅い（数十秒） |
| **AIO / GGUF** | `set_adapters()` で動的切り替え | 即時 |
| **ComfyUI (AIO)** | ワークフロー内 `LoraLoaderModelOnly` ノード（最大3個） | ComfyUI依存 |
| **ComfyUI (Nunchaku)** | `NunchakuQwenImageLoraStackV3` ノード（最大10個） | ComfyUI依存 |
| **ComfyUI (GGUF)** | ワークフロー内 `LoraLoaderModelOnly` ノード（最大3個） | ComfyUI依存 |

Nunchaku版では、LoRAの有効/無効や強度を変更すると、次の生成開始時にパイプラインが自動的にリロードされます。同じ構成のまま連続生成する場合はリロードは発生しません。

非互換なLoRAファイルはスキップされ、サーバーはクラッシュしません。

---

## ギャラリーモード

`--gallery` オプションで有効化します。生成履歴の閲覧・再利用が可能になります。

### 機能一覧

- **履歴表示**: 過去の生成結果がタイムスタンプ・プロンプト・シード値とともに一覧表示
- **サムネイル**: 入力画像と結果画像がサムネイルで表示。クリックで拡大（描画エディタで開く）
- **ダウンロード**: 各画像に DL リンク
- **画像再利用**: 各サムネイル下の **Img1** / **Img2** ラジオボタンで、履歴の画像を次の生成の入力に設定
  - 選択するとファイル入力欄下にサムネイル + IDが小さく表示される
  - t2i モードは自動的にOFFになる
- **履歴削除**: 自分の生成履歴は `×` ボタンで削除可能（他ユーザーの履歴は削除不可）。削除後もプレースホルダーとして表示される
- **自動クリーンアップ**: 1時間以上経過した一時ファイルは5分ごとに自動削除

### 画像入力の選択ルール

ファイルアップロードとギャラリー選択が競合する場合の動作:

| 操作 | 結果 |
|------|------|
| Img1/Img2 ラジオボタンをクリック | 対応するファイル入力がクリアされる |
| ファイルをアップロード（選択 or D&D） | 対応するギャラリー選択が解除される |
| アップロード済みファイルがある場合 | ファイルが優先される |

---

## 描画エディタ

画像をクリックすると全画面の描画エディタが開きます。サムネイル、アップロードプレビュー、生成結果のいずれもクリックで開けます。

### エディタの構造

- **背景レイヤー**: 元の画像（変更不可）
- **オーバーレイレイヤー**: 描画内容（透明レイヤー上に描く）

2つのレイヤーが重なって表示され、描画はすべてオーバーレイレイヤーに対して行われます。

### 描画ツール

| ツール | ボタン | 説明 |
|--------|--------|------|
| **Pen** | `Pen` | 通常のフリーハンドペン。固定太さ |
| **Brush** | `Brush` | 筆圧対応ペン。ペンタブの筆圧に応じて線の太さが変化 |
| **Airbrush** | `Air` | 筆圧対応エアブラシ。筆圧で散布範囲と濃度が変化 |
| **Eraser** | `Eraser` | 消しゴム。オーバーレイの描画のみを消去（元画像は消えない） |
| **Cover** | `Cover` | カバー。白色で元画像を塗りつぶす（オーバーレイに白を描画） |
| **Select** | `Select` | 範囲選択。コピー＆ペーストに使用 |

> **筆圧について**: Brush と Airbrush はペンタブレット等のポインターデバイスの筆圧に対応しています。マウスの場合は固定圧力（0.5）で動作します。

### カラーパレット

- 10色のプリセットカラー（黒、白、赤、オレンジ、黄、緑、青、紫、ピンク、茶）
- カスタムカラーピッカー（任意の色を選択可能）

### 線の太さ

7段階: **1** / **2** / **4** / **8** / **14** / **24** / **64** ピクセル

### ルーペ（拡大表示）

描画中、カーソル付近に4倍拡大のルーペが自動表示されます。

- 円形のルーペがカーソルの右上に追従
- 画面端では自動的に位置が調整される
- 中央に十字線が表示される
- Select / Paste モード中は非表示

### 操作ボタン

| ボタン | 説明 |
|--------|------|
| **Undo** | 直前の操作を取り消し（最大30ステップ） |
| **Pause** | 描画を一時保存してエディタを閉じる。DRAFTラベル付きサムネイルとして保存される |
| **Save(+bg)** | 背景 + オーバーレイを合成して保存 |
| **Save(line)** | オーバーレイのみを保存（背景なし、透過PNG） |
| **Close** | エディタを閉じる（Escキーでも可） |

### コピー＆ペースト

1. **Select** ツールに切り替え
2. ドラッグで範囲を選択（緑の点線枠）
3. コピーメニューが表示される:
   - **Copy (bg)**: 背景レイヤーのみコピー
   - **Copy (draw)**: オーバーレイのみコピー
   - **Copy (+bg)**: 合成画像をコピー
4. **Paste** ボタンが有効になる
5. クリックするとペーストモードに入る:
   - ドラッグで移動
   - 四隅をドラッグでリサイズ
   - **Lock ratio** チェックボックスをONにすると、アスペクト比を固定したままリサイズできる
   - **Confirm** で確定 / **Cancel** で取り消し（Enterキー / Escキーでも可）

### ドラフト（一時保存）

**Pause** ボタンで描画途中の状態を保存できます。

- DRAFTラベル付きのサムネイルとして表示される
- クリックで描画を再開（背景 + オーバーレイが復元される）
- ドラフトはImg1/Img2としての使用は不可

### 保存した描画の使い方

保存された描画（composite / line-only）はサムネイルとして表示され:

- **DL** リンクでダウンロード
- **Del** ボタンで削除
- **Img1** / **Img2** ラジオボタンで次の生成の入力として使用可能

### 白紙キャンバス

フォーム内の **+ Blank Sketch** ボタンで、元画像なしの白紙キャンバス（1024x1024）で描画を開始できます。

---

## ポーリングの安全機能

生成中のステータス確認（ポーリング）には安全停止機能があります:

| 条件 | 動作 |
|------|------|
| 2分以上応答なし | タイムアウトエラーを表示してポーリング停止 |
| パスワードエラーが3回連続 | 認証エラーを表示してポーリング停止 |

---

## モデル情報表示

フォーム上部にロード済みモデルの情報が表示されます:

- Pipeline / Transformer / Text Encoder / Tokenizer / VAE / Dtype / Steps / LoRA

---

## エラーメッセージ

エラーメッセージは日本語・英語の2言語で表示されます。

---

## キーボードショートカット

| キー | 描画エディタでの動作 |
|------|---------------------|
| `Esc` | エディタを閉じる / ペーストをキャンセル |
| `Enter` | ペーストを確定 |

---

# Web Server User Guide

Feature overview and usage instructions for the AI image editing web server.

## Server Types

| Server | Script | Model | Features |
|--------|--------|-------|----------|
| **Nunchaku** | `app_nunchaku.py` | Qwen-Image-Edit-2509 Lightning | Fast with SVDQ quantization. Runs on RTX 3xxx (12GB) |
| **GGUF** | `app_gguf.py` | Qwen-Image-Edit-Rapid-AIO-V23 (GGUF) | Selectable quantization levels from Q3_K to Q8_0. Low VRAM friendly |
| **AIO** | `app_aio.py` | Qwen-Image-Edit-Rapid-AIO-V23 | bf16 full precision. Requires more VRAM |
| **ComfyUI (AIO)** | `app_comfyui.py` | Via ComfyUI API (AIO workflow) | No torch/diffusers dependency. Requires separate ComfyUI instance |
| **ComfyUI (Nunchaku)** | `app_comfyui_nunchaku.py` | Via ComfyUI API (Nunchaku workflow) | No torch/diffusers dependency. Requires separate ComfyUI instance |
| **ComfyUI (GGUF)** | `app_comfyui_gguf.py` | Via ComfyUI API (GGUF workflow) | No torch/diffusers dependency. Uses GGUF quantized models |

## Getting Started

```powershell
cd server

# Basic startup (password: password, port: 5000)
python app_nunchaku.py

# Specify password and port
python app_nunchaku.py --password mysecret --port 8080

# Enable gallery mode (history and sharing)
python app_nunchaku.py --gallery --password mysecret

# Specify LoRA (multiple allowed)
python app_nunchaku.py --lora "path/to/lora.safetensors"
python app_nunchaku.py --lora "repo_id::weight.safetensors" --lora "another.safetensors"

# Prompt presets
python app_nunchaku.py --preset "Enhance::Enhance quality." --preset "Remove text::Remove all text."

# Expose to LAN (accessible from other devices)
python app_nunchaku.py --host 0.0.0.0
```

### ComfyUI Server Startup

ComfyUI servers have no torch/diffusers dependency and delegate inference to a running ComfyUI instance via API.

```powershell
cd server

# ComfyUI (AIO workflow)
python app_comfyui.py
python app_comfyui.py --password mysecret --port 5000
python app_comfyui.py --comfyui-url http://192.168.1.100:8188
python app_comfyui.py --comfyui-path D:\ComfyUI    # Auto-register LoRA path + reboot ComfyUI

# ComfyUI (Nunchaku workflow)
python app_comfyui_nunchaku.py
python app_comfyui_nunchaku.py --password mysecret --port 5000
python app_comfyui_nunchaku.py --comfyui-url http://192.168.1.100:8188
python app_comfyui_nunchaku.py --comfyui-path D:\ComfyUI

# ComfyUI (GGUF workflow)
python app_comfyui_gguf.py
python app_comfyui_gguf.py --password mysecret --port 5000
python app_comfyui_gguf.py --comfyui-url http://192.168.1.100:8188
python app_comfyui_gguf.py --comfyui-path D:\ComfyUI
```

### Startup Options

#### Common Options
| Option | Description | Default |
|--------|-------------|---------|
| `--host HOST` | Bind host | `127.0.0.1` |
| `--port PORT` | Bind port | `5000` |
| `--password PW` | Generation password | `password` |
| `--no-progress` | Hide HF download progress | Shown |
| `--no-offload` | Disable offloading (high VRAM) | Auto |
| `--lora REPO_OR_PATH` | LoRA weights (repeatable). `path` or `repo_id::weight_name` | None |
| `--gallery` | Enable gallery mode | Disabled |
| `--preset "label::prompt"` | Preset button (repeatable) | None |

#### Server-specific Options
| Option | Target | Description |
|--------|--------|-------------|
| `--steps N` | Nunchaku | Inference steps: 4 or 8 (default: 8) |
| `--rank N` | Nunchaku | Nunchaku rank: 32 or 128 |
| `--num-blocks-on-gpu N` | Nunchaku | Number of blocks to keep on GPU in low-VRAM mode |
| `--gguf-file PATH` | GGUF | GGUF file within HF repository |
| `--gguf-local PATH` | GGUF | Use local GGUF file directly |
| `--offload` | AIO | Sequential CPU offload (low VRAM) |
| `--comfyui-url URL` | ComfyUI both | ComfyUI API URL (default: `http://127.0.0.1:8188`) |
| `--comfyui-path DIR` | ComfyUI both | ComfyUI installation directory. Auto-registers LoRA path and reboots ComfyUI |
| `--steps N` | ComfyUI both | Inference steps (default: 8) |
| `--cfg N` | ComfyUI both | CFG scale (default: 1.0) |

> **Note:** The GGUF server does not support `--offload` (incompatible with GGUF tensors).
> **Note:** The AIO server requires significantly more VRAM than GGUF (bf16 full precision).
> **Note:** ComfyUI servers require a separate running ComfyUI instance. Connection is verified at startup with a 60-second retry.
> **Note:** ComfyUI (Nunchaku) requires the Nunchaku library and custom nodes `nunchaku-ai/ComfyUI-nunchaku` and `ussoewwin/ComfyUI-QwenImageLoraLoader`.
> **Note:** ComfyUI (GGUF) requires the gguf library and custom node `city96/ComfyUI-GGUF`.

---

## Basic Operations

### 1. Login

With gallery mode (`--gallery`), a password input screen is shown first. Enter the password specified at startup to log in.

Without gallery mode, the form is displayed directly (password is entered within the form).

### 2. Image Editing (img2img)

1. Enter the password in the **Password** field
2. Upload the image to edit in **Image 1** (file picker or drag & drop)
3. Optionally add a reference image in **Image 2 / REF**
4. Enter editing instructions in **Prompt** (default prompt used if empty)
5. Select input image size in **Pre-resize**. Nunchaku/GGUF/AIO: 0.3M / 1M pixels; ComfyUI: 1M / 2M pixels
6. Click **Generate**

### 3. Text-to-Image (t2i)

1. Check the **Text-to-Image (t2i) mode** checkbox
2. Image input area is hidden
3. Enter a description of the image to generate in **Prompt** (required)
4. Click **Generate** — a 1024x1024 image is generated

### 4. Progress and Cancellation

- Step progress is shown in real time during generation (e.g. `Processing... (step 3/8)`)
- A progress bar provides visual feedback
- Click **Cancel** to abort generation
- Job queue: 1 processing + 2 waiting. Submitting 3+ jobs simultaneously returns `503 BUSY`
- **Preview display** (ComfyUI only): "Show preview during generation" checkbox shows real-time preview images during generation. Requires ComfyUI Settings → Preview Method enabled

---

## Prompt Features

### Preset Buttons

Presets configured with `--preset` appear as buttons above the prompt field.

- Click to fill the prompt with the preset text
- A confirmation dialog is shown if the prompt field already has text
- Example: `--preset "Enhance::Enhance quality."` creates an "Enhance" button

### Prompt Clear

The **x** button next to the prompt label clears all prompt text.

### Prompt Translation

Translation buttons below the prompt field translate the current prompt.

| Button | Function |
|--------|----------|
| `-> EN` | Translate to English |
| `-> ZH` | Translate to Chinese |
| `-> JA` | Translate to Japanese |

Translation is provided because some models produce better results with English or Chinese prompts.

---

## Continuous Mode

Check the **Continuous mode** checkbox to automatically set the generated image as Img1 when generation completes.

This makes iterative editing workflows easy — edit the result repeatedly by changing prompts.

The result area also shows **Img1** / **Img2** radio buttons, allowing you to immediately use the result as input for the next generation without waiting for the gallery to refresh.

---

## LoRA

### Configuration

**Nunchaku / GGUF / AIO:** Specify with the `--lora` option at startup (repeatable).

```powershell
# Local file
python app_nunchaku.py --lora "path/to/lora.safetensors"

# HuggingFace repository
python app_nunchaku.py --lora "repo_id::weight_name.safetensors"

# Multiple LoRAs
python app_nunchaku.py --lora "lora1.safetensors" --lora "lora2.safetensors"
```

**ComfyUI:** Place LoRA files in the `server/LoRA/` folder. Use `--comfyui-path` to auto-register the path in `extra_model_paths.yaml`. No `--lora` option needed.

### WebUI Controls

- Registered LoRAs appear as checkboxes + strength sliders (0.0 to 2.0) in the form
- All are unchecked by default
- Toggle on/off and adjust strength per generation

### Behavior by Server

| Server | LoRA Switching Method | Speed |
|--------|----------------------|-------|
| **Nunchaku** | Full pipeline reload (only when config changes) | Slow (tens of seconds) |
| **AIO / GGUF** | Dynamic switching via `set_adapters()` | Instant |
| **ComfyUI (AIO)** | `LoraLoaderModelOnly` nodes in workflow (max 3) | ComfyUI dependent |
| **ComfyUI (Nunchaku)** | `NunchakuQwenImageLoraStackV3` node (max 10) | ComfyUI dependent |
| **ComfyUI (GGUF)** | `LoraLoaderModelOnly` nodes in workflow (max 3) | ComfyUI dependent |

In the Nunchaku version, changing LoRA enable/disable or strength triggers an automatic pipeline reload before the next generation. Consecutive generations with the same config skip the reload.

Incompatible LoRA files are skipped and the server does not crash.

---

## Gallery Mode

Enable with the `--gallery` option. Allows browsing and reusing generation history.

### Features

- **History**: Past results listed with timestamps, prompts, and seed values
- **Thumbnails**: Input and result images shown as thumbnails. Click to enlarge (opens in drawing editor)
- **Download**: DL link on each image
- **Image Reuse**: **Img1** / **Img2** radio buttons below each thumbnail to set history images as input for new generations
  - A thumbnail + ID is shown below the file input when selected
  - t2i mode is automatically turned OFF
- **History Deletion**: Delete your own history entries with the **x** button (cannot delete other users' entries). Deleted entries remain as placeholders
- **Auto-cleanup**: Temporary files older than 1 hour are automatically deleted every 5 minutes

### Image Input Priority

When file uploads and gallery selections conflict:

| Action | Result |
|--------|--------|
| Click Img1/Img2 radio button | Corresponding file input is cleared |
| Upload a file (picker or D&D) | Corresponding gallery selection is cleared |
| File already uploaded | File takes priority |

---

## Drawing Editor

Click any image to open the full-screen drawing editor. Gallery thumbnails, upload previews, and generated results can all be clicked to open.

### Editor Structure

- **Background layer**: Original image (read-only)
- **Overlay layer**: Drawing content (drawn on a transparent layer)

The two layers are displayed stacked, and all drawing is done on the overlay layer.

### Drawing Tools

| Tool | Button | Description |
|------|--------|-------------|
| **Pen** | `Pen` | Standard freehand pen with fixed width |
| **Brush** | `Brush` | Pressure-sensitive pen. Line width varies with pen tablet pressure |
| **Airbrush** | `Air` | Pressure-sensitive airbrush. Spray radius and density vary with pressure |
| **Eraser** | `Eraser` | Eraser. Removes overlay drawings only (original image is not affected) |
| **Cover** | `Cover` | Cover. Paints white over the original image (draws white on the overlay) |
| **Select** | `Select` | Region selection. Used for copy & paste |

> **About pressure**: Brush and Airbrush respond to pointer device pressure (pen tablets, etc.). With a mouse, they operate at fixed pressure (0.5).

### Color Palette

- 10 preset colors (black, white, red, orange, yellow, green, blue, purple, pink, brown)
- Custom color picker (any color)

### Line Sizes

7 levels: **1** / **2** / **4** / **8** / **14** / **24** / **64** pixels

### Loupe (Magnifier)

A 4x magnification loupe automatically appears near the cursor while drawing.

- Circular loupe follows the cursor at the upper right
- Automatically repositioned near screen edges
- Crosshair displayed at center
- Hidden during Select / Paste mode

### Action Buttons

| Button | Description |
|--------|-------------|
| **Undo** | Undo the last action (up to 30 steps) |
| **Pause** | Save drawing in progress and close the editor. Saved as a DRAFT-labeled thumbnail |
| **Save(+bg)** | Save composite of background + overlay |
| **Save(line)** | Save overlay only (no background, transparent PNG) |
| **Close** | Close the editor (Esc key also works) |

### Copy & Paste

1. Switch to the **Select** tool
2. Drag to select a region (green dashed outline)
3. A copy menu appears:
   - **Copy (bg)**: Copy background layer only
   - **Copy (draw)**: Copy overlay only
   - **Copy (+bg)**: Copy composite image
4. The **Paste** button becomes active
5. Click to enter paste mode:
   - Drag to move
   - Drag corners to resize
   - Check **Lock ratio** to resize while maintaining the aspect ratio
   - **Confirm** to apply / **Cancel** to discard (Enter / Esc keys also work)

### Draft (Temporary Save)

The **Pause** button saves the current drawing state.

- Displayed as a DRAFT-labeled thumbnail
- Click to resume drawing (background + overlay are restored)
- Drafts cannot be used as Img1/Img2

### Using Saved Drawings

Saved drawings (composite / line-only) appear as thumbnails with:

- **DL** link to download
- **Del** button to delete
- **Img1** / **Img2** radio buttons to use as input for the next generation

### Blank Canvas

The **+ Blank Sketch** button in the form starts drawing on a blank white canvas (1024x1024) without a source image.

---

## Polling Safety

Status polling during generation has automatic safety stops:

| Condition | Action |
|-----------|--------|
| No response for 2+ minutes | Shows timeout error and stops polling |
| 3 consecutive password errors | Shows auth error and stops polling |

---

## Model Info Display

Loaded model information is shown at the top of the form:

- Pipeline / Transformer / Text Encoder / Tokenizer / VAE / Dtype / Steps / LoRA

---

## Error Messages

Error messages are displayed in both Japanese and English.

---

## Keyboard Shortcuts

| Key | Drawing Editor Action |
|-----|----------------------|
| `Esc` | Close editor / Cancel paste |
| `Enter` | Confirm paste |
