chcp 65001 >nul

rem Connects to a running ComfyUI instance.
rem To use local LoRA folder, add: --comfyui-path "C:\path\to\ComfyUI"
rem Without --comfyui-path, LoRAs are discovered from ComfyUI automatically.

python app_comfyui_gguf.py --host 0.0.0.0 --port 18188 --password password --steps 8 --gallery^
 --comfyui-url "http://127.0.0.1:8188"^
 --preset "画像1に画像2の服を着せる::将 Image-1 中的角色穿上 Image-2 的服装"^
 --preset "高画質化::Enhance quality."^
 --preset "テキスト除去::Remove all text."^
 --preset "LoRA_PAINT::Color this panelpainter"^
 --preset "LoRA_REAL::transform the image into high quality realistic photograph. [female|male]"

pause
