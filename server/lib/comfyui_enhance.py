"""Shared ComfyUI prompt enhancement helper.

Runs the enhance_prompt_api.json workflow on ComfyUI to expand
a short prompt into a detailed one using a local VLM (QwenVL GGUF).
"""

from __future__ import annotations

import copy
import io
import time
import uuid


def _upload_or_placeholder(upload_fn, image_bytes: bytes | None, suffix: str) -> str:
    """Upload image bytes or a white placeholder, return stored name."""
    if image_bytes:
        name = f"enhance_{uuid.uuid4().hex[:8]}{suffix}.png"
        return upload_fn(image_bytes, name)
    from PIL import Image
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    buf = io.BytesIO()
    try:
        img.save(buf, format="PNG")
        name = f"enhance_{uuid.uuid4().hex[:8]}{suffix}_blank.png"
        return upload_fn(buf.getvalue(), name)
    finally:
        buf.close()
        img.close()


def run_enhance(
    upload_fn,
    submit_fn,
    get_history_fn,
    workflow_template: dict,
    prompt_text: str,
    image_bytes: bytes | None = None,
    image2_bytes: bytes | None = None,
    timeout: float = 360,
) -> str:
    """Run enhance prompt workflow on ComfyUI and return enhanced text.

    Args:
        upload_fn: comfyui_upload_image(image_bytes, filename) -> stored_name
        submit_fn: comfyui_submit_prompt(workflow) -> prompt_id
        get_history_fn: comfyui_get_history(prompt_id) -> dict
        workflow_template: Parsed enhance workflow JSON
        prompt_text: User's short prompt to enhance
        image_bytes: Optional input image bytes (PNG/JPEG) for Img1
        image2_bytes: Optional input image bytes (PNG/JPEG) for Img2
        timeout: Max wait time in seconds

    Returns:
        Enhanced prompt text
    """
    wf = copy.deepcopy(workflow_template)

    # Set prompt text (node 35: PrimitiveStringMultiline)
    wf["35"]["inputs"]["value"] = prompt_text

    # Handle image inputs (node 12: Img1, node 59: Img2)
    wf["12"]["inputs"]["image"] = _upload_or_placeholder(upload_fn, image_bytes, "")
    wf["59"]["inputs"]["image"] = _upload_or_placeholder(upload_fn, image2_bytes, "_img2")

    # Submit workflow
    prompt_id = submit_fn(wf)

    # Poll for completion
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(2)
        hist = get_history_fn(prompt_id)
        if prompt_id not in hist:
            continue

        entry = hist[prompt_id]

        # Check for errors
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            msgs = status.get("messages", [])
            raise RuntimeError(f"ComfyUI workflow error: {msgs}")

        outputs = entry.get("outputs", {})
        if not outputs:
            continue

        # Extract text from PreviewAny (node 33)
        text = _extract_text(outputs, "33")
        if text:
            return text

        # Fallback: scan all nodes
        for nid in outputs:
            text = _extract_text(outputs, nid)
            if text:
                return text

        raise RuntimeError(
            f"Workflow completed but no text output found. "
            f"Output nodes: {list(outputs.keys())}"
        )

    raise TimeoutError(f"Enhance prompt timed out after {timeout}s")


def _extract_text(outputs: dict, node_id: str) -> str | None:
    """Try to extract text from a node's output in ComfyUI history."""
    node_out = outputs.get(node_id, {})
    for key in ("text", "string", "preview_text", "any"):
        val = node_out.get(key)
        if val is None:
            continue
        if isinstance(val, list) and val:
            return str(val[0]).strip()
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None
