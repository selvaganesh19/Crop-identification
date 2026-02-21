import os
import gradio as gr
from dotenv import load_dotenv
from PIL import Image
import base64
import io
import hashlib
import traceback
from openai import AzureOpenAI

load_dotenv()

# =====================================
# AZURE CONFIG
# =====================================
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv(
        "AZURE_OPENAI_API_VERSION",
        "2024-02-15-preview"
    ),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# =====================================
# SESSION MEMORY (API SAFE ✅)
# =====================================
user_sessions = {}
crop_cache = {}


def get_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()


# =====================================
# 🌾 IDENTIFY CROP
# =====================================
def identify_crop(image_file, session_id):

    if image_file is None:
        return "❌ Upload crop image."

    try:
        img = Image.open(image_file)

        if img.mode != "RGB":
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")

        image_bytes = buffer.getvalue()
        image_hash = get_hash(image_bytes)

        # ✅ IMAGE CACHE
        if image_hash in crop_cache:
            result = crop_cache[image_hash]
            user_sessions[session_id] = result
            return result

        image_base64 = base64.b64encode(image_bytes).decode()

        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content":
                    "Identify crop only. No greeting."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text":
                            "Return strictly:\n"
                            "Crop Name:\n"
                            "Description:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":
                                f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )

        result = response.choices[0].message.content

        crop_cache[image_hash] = result
        user_sessions[session_id] = result

        return result

    except Exception:
        return traceback.format_exc()


# =====================================
# 💬 CHATBOT
# =====================================
def chat_ui(message, history, session_id):

    if history is None:
        history = []

    crop_info = user_sessions.get(session_id)

    if not crop_info:
        reply = "⚠️ Upload and identify crop first."
    else:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content":
                    "You are a farming advisor."
                },
                {
                    "role": "user",
                    "content":
                    f"{crop_info}\n\nQuestion:{message}"
                }
            ],
            max_tokens=500
        )

        reply = response.choices[0].message.content

    history.append([message, reply])

    return history, ""


# =====================================
# UI (FOR HF API ENDPOINTS)
# =====================================
with gr.Blocks() as demo:

    session_id = gr.Textbox(visible=False)

    gr.Markdown("# 🌾 Crop Identification API")

    image_input = gr.Image(type="filepath")
    identify_btn = gr.Button("Identify Crop")
    image_output = gr.Textbox()

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    send = gr.Button("Send")

    identify_btn.click(
        identify_crop,
        [image_input, session_id],
        image_output,
        api_name="/identify_crop"
    )

    send.click(
        chat_ui,
        [msg, chatbot, session_id],
        [chatbot, msg],
        api_name="/chat_ui"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)