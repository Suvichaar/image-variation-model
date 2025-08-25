import os
import io
import re
import json
import time
import base64
import zipfile
import random
import string
from io import BytesIO
from datetime import datetime

import requests
import boto3
from PIL import Image
import streamlit as st

# ============================
# Minimal app: Upload ONE image → GPT-4o caption → 5–6 DALL·E images inspired by it
# - Uploads originals to S3
# - Returns CDN (resized) URLs and a ZIP download
# ============================

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Suvichaar: Image Variations", page_icon="🎨", layout="centered")
st.title("🎨 Suvichaar — Image Variations from a Reference")
st.caption("Upload ONE reference image → GPT-4o generates a rich caption → we synthesize 5–6 stylistic variations with DALL·E → upload to S3 → show CDN links")

# ---------------------------
# Secrets / Config
# ---------------------------
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default

# Azure OpenAI (GPT-4o for vision caption + prompt safety)
AZURE_API_KEY     = get_secret("AZURE_API_KEY")
AZURE_ENDPOINT    = get_secret("AZURE_ENDPOINT")  # https://<resource>.openai.azure.com
AZURE_DEPLOYMENT  = get_secret("AZURE_DEPLOYMENT", "gpt-4o")  # vision-capable deployment
AZURE_API_VERSION = get_secret("AZURE_API_VERSION", "2024-08-01-preview")

# Azure DALL·E (Images)
DALE_ENDPOINT     = get_secret("DALE_ENDPOINT")  # e.g. https://.../openai/deployments/dall-e-3/images/generations?api-version=2024-02-01
DAALE_KEY         = get_secret("DAALE_KEY")

# AWS S3
AWS_ACCESS_KEY        = get_secret("AWS_ACCESS_KEY")
AWS_SECRET_KEY        = get_secret("AWS_SECRET_KEY")
AWS_SESSION_TOKEN     = get_secret("AWS_SESSION_TOKEN")  # optional
AWS_REGION            = get_secret("AWS_REGION", "ap-south-1")
AWS_BUCKET            = get_secret("AWS_BUCKET", "suvichaarapp")
S3_PREFIX             = get_secret("S3_PREFIX", "media")  # used for images

# CDN image handler prefix (base64-encoded template handled by your Serverless Image Handler)
CDN_PREFIX_MEDIA  = get_secret("CDN_PREFIX_MEDIA", "https://media.suvichaar.org/")

# Sanity checks (warn if missing)
missing = []
for k in ["AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_DEPLOYMENT", "DALE_ENDPOINT", "DAALE_KEY", "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_BUCKET"]:
    if not get_secret(k, None):
        missing.append(k)
if missing:
    st.warning("Add these secrets in `.streamlit/secrets.toml`: " + ", ".join(missing))

# ---------------------------
# AWS helpers (client)
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_s3_client():
    kwargs = dict(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
    if AWS_SESSION_TOKEN:
        kwargs["aws_session_token"] = AWS_SESSION_TOKEN
    return boto3.client("s3", **kwargs)


def build_resized_cdn_url(bucket: str, key_path: str, width: int, height: int) -> str:
    template = {
        "bucket": bucket,
        "key": key_path,
        "edits": {"resize": {"width": width, "height": height, "fit": "cover"}},
    }
    encoded = base64.urlsafe_b64encode(json.dumps(template).encode()).decode()
    return f"{CDN_PREFIX_MEDIA}{encoded}"

# ---------------------------
# Prompt helpers
# ---------------------------
def _variation_token(k=8) -> str:
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))

STYLES = [
    "minimalist geometric vector, soft gradients",
    "flat illustration, bold shapes, high contrast",
    "pastel vector, friendly mascots, rounded edges",
    "clean infographic style, iconography, airy spacing",
    "playful educational poster vibe, simple shapes",
    "modern editorial illustration, crisp lines"
]


def sanitize_prompt(chat_url: str, headers: dict, original_prompt: str) -> str:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Rewrite image prompts to be safe, positive, inclusive, and family-friendly. "
                    "Remove any hate/harassment/violence/adult/illegal/extremist content, slogans, logos, "
                    "or real-person likenesses. Keep the core idea and vector-art style. Return ONLY the rewritten prompt."
                ),
            },
            {"role": "user", "content": f"Original prompt:\n{original_prompt}\n\nRewritten safe prompt:"},
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }
    try:
        r = requests.post(chat_url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return (
        original_prompt
        + " — flat vector illustration, clean geometric shapes, smooth gradients; family-friendly; no text, no logos, no watermarks, no real-person likeness."
    )


# ---------------------------
# Vision caption with GPT-4o
# ---------------------------

def b64_image(file_bytes: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(file_bytes).decode()}"


def caption_image_with_gpt(file_bytes: bytes, target_lang: str = "en") -> str:
    """
    Ask GPT-4o to describe the uploaded image as a single, rich prompt.
    Uses Azure OpenAI (Vision) → returns a caption string.
    """
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"

    system_msg = (
        "You are an art director. Describe the uploaded image as ONE concise prompt for vector illustration generation. "
        "Focus on subjects, setting, composition, perspective/camera, lighting, color palette, mood, and style hints. "
        "Keep it family-friendly and avoid brand names, logos, text, or celebrity likeness. "
        f"Write the description in {target_lang}."
    )

    # Encode the image as base64 → send via `image_url`
    b64_img = base64.b64encode(file_bytes).decode()
    img_payload = f"data:image/png;base64,{b64_img}"

    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Describe the image in {target_lang} as one short paragraph."},
                {"type": "image_url", "image_url": {"url": img_payload}},  # <-- FIXED HERE ✅
            ],
        },
    ]

    payload = {"messages": messages, "temperature": 0.3, "max_tokens": 500}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        else:
            st.error(f"GPT-4o captioning error {r.status_code}: {r.text[:300]}")
            return "An abstract, colorful, vector-style illustration inspired by the uploaded image."
    except Exception as e:
        st.error(f"GPT-4o caption request failed: {e}")
        return "An abstract, colorful, vector-style illustration inspired by the uploaded image."



# ---------------------------
# DALL·E generation
# ---------------------------

def dalle_generate(prompt: str) -> bytes | None:
    headers = {"Content-Type": "application/json", "api-key": DAALE_KEY}
    payload = {"prompt": prompt, "n": 1, "size": "1024x1024"}
    try:
        r = requests.post(DALE_ENDPOINT, headers=headers, json=payload, timeout=180)
        if r.status_code == 200:
            image_url = r.json()["data"][0]["url"]
            img = requests.get(image_url, timeout=180).content
            return img
        elif r.status_code in (400, 403):
            # blocked → minimal fallback
            fallback = (
                "Abstract geometric illustration of the same concept, flat vector style, bright harmonious colors, "
                "clean shapes; no text or logos; family-friendly."
            )
            r2 = requests.post(DALE_ENDPOINT, headers=headers, json={"prompt": fallback, "n": 1, "size": "1024x1024"}, timeout=180)
            if r2.status_code == 200:
                image_url = r2.json()["data"][0]["url"]
                return requests.get(image_url, timeout=180).content
            return None
        else:
            st.info(f"DALL·E error {r.status_code}: {r.text[:200]}")
            return None
    except Exception as e:
        st.info(f"DALL·E request failed: {e}")
        return None


# ---------------------------
# UI — Upload + Controls
# ---------------------------

uploaded = st.file_uploader("Upload ONE reference image", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False)
colA, colB = st.columns(2)
num_outputs = colA.slider("How many images?", 5, 6, 6)
lang_choice = colB.selectbox("Caption language", ["English (en)", "Hindi (hi)"], index=0)

run = st.button("🚀 Generate Variations")

# ---------------------------
# Main flow
# ---------------------------
if run:
    if not uploaded:
        st.error("Please upload one image.")
        st.stop()

    file_bytes = uploaded.getvalue()
    try:
        st.image(Image.open(BytesIO(file_bytes)), caption="Reference", use_container_width=True)
    except Exception:
        st.info("Preview not available; continuing…")

    target_lang = "en" if lang_choice.startswith("English") else "hi"

    with st.spinner("Describing the reference with GPT-4o…"):
        base_caption = caption_image_with_gpt(file_bytes, target_lang)

    # Safety pass
    chat_headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    chat_url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    safe_base = sanitize_prompt(chat_url, chat_headers, base_caption)

    # Build N prompts with style variations
    chosen_styles = STYLES[:]
    random.shuffle(chosen_styles)

    prompts = []
    for i in range(num_outputs):
        style = chosen_styles[i % len(chosen_styles)]
        prompts.append(
            f"{safe_base}\n\nStyle: {style}. Unique variation code: {_variation_token()}. Vector art, no text/logos/watermarks."
        )

    s3 = get_s3_client()
    out_urls = []
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        progress = st.progress(0.0, text="Generating images…")
        slug = re.sub(r"[^a-z0-9\-]+", "-", (uploaded.name.split(".")[0].lower() or "ref")).strip("-")[:64] or "ref"
        for i, p in enumerate(prompts, start=1):
            img_bytes = dalle_generate(p)
            if not img_bytes:
                st.error(f"Image {i}: generation failed.")
                progress.progress(i / num_outputs)
                continue

            # Upload to S3
            key = f"{S3_PREFIX.rstrip('/')}/image-variations/{slug}/img_{i}.jpg"
            try:
                s3.upload_fileobj(BytesIO(img_bytes), AWS_BUCKET, key, ExtraArgs={"ContentType": "image/jpeg"})
                cdn = build_resized_cdn_url(AWS_BUCKET, key, 720, 1200)
                out_urls.append((i, cdn))
            except Exception as e:
                st.error(f"Upload failed for image {i}: {e}")

            # Add to ZIP as well
            z.writestr(f"{slug}_variation_{i}.jpg", img_bytes)
            progress.progress(i / num_outputs, text=f"Generating images… ({i}/{num_outputs})")
        progress.empty()

    # Show gallery
    if out_urls:
        st.success("Done! Here are your CDN image URLs (resized 720x1200):")
        cols = st.columns(3)
        for idx, url in out_urls:
            with cols[(idx - 1) % 3]:
                st.image(url, caption=f"Variation {idx}")
                st.code(url, language="text")

        # ZIP download
        zip_buf.seek(0)
        st.download_button(
            "⬇️ Download all originals (ZIP)",
            data=zip_buf.getvalue(),
            file_name=f"{slug}_variations.zip",
            mime="application/zip",
        )
    else:
        st.error("No images were generated.")
