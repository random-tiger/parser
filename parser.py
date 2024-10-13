import streamlit as st
import os
import base64
from io import BytesIO
from openai import OpenAI
from pdf2image import convert_from_path
import pandas as pd
import numpy as np
import faiss
import concurrent.futures
from sklearn.preprocessing import normalize

# Pull OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

# Set the models and API key
CHAT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to encode image from PIL Image object
def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to process a single page
def process_page(args):
    page_num, image = args
    base64_image = encode_image_from_pil(image)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ""},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.0,
    )

    assistant_reply = response.choices[0].message.content

    return page_num, assistant_reply

# Rest of the code remains the same
