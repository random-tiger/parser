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

# Define the path to the Poppler binaries
poppler_path = os.path.join(os.getcwd(), "poppler_binaries", "bin")

# Convert PDF to images using the local Poppler binaries
images = convert_from_path("temp.pdf", poppler_path=poppler_path)

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

# Streamlit app interface
st.title("PDF Document Analyzer with OpenAI and FAISS")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Convert PDF to images
    st.write("Processing the uploaded PDF...")
    pdf_bytes = uploaded_file.read()

    with open("temp.pdf", "wb") as f:
        f.write(pdf_bytes)
    
    images = convert_from_path("temp.pdf")

    # Prepare list of pages
    pages = list(enumerate(images, start=1))

    # Process pages concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(process_page, pages))

    # Extract page numbers and descriptions
    page_nums, descriptions = zip(*results)

    # Create a DataFrame with the page descriptions
    data = {
        'page_num': page_nums,
        'description': descriptions
    }

    df = pd.DataFrame(data)

    st.write("Document analysis completed:")
    st.dataframe(df)

    # Function to get embeddings (unchanged)
    def get_embedding(text, model=EMBEDDING_MODEL):
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding

    # Concurrently get embeddings for descriptions
    def process_embeddings(df, max_workers=10):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            embeddings = list(executor.map(get_embedding, df['description'].tolist()))
        return embeddings

    # Apply concurrency to generate embeddings
    df['embedding'] = process_embeddings(df)

    # Convert embeddings to a NumPy array
    embeddings = np.array(df['embedding'].tolist()).astype('float32')

    # Normalize embeddings
    normalized_embeddings = normalize(embeddings, norm='l2')

    # Faiss index creation
    d = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # Using Inner Product for cosine similarity
    index.add(normalized_embeddings)

    # Search function
    def search_embeddings(query, index, df, n=3, model=EMBEDDING_MODEL):
        query_embedding = get_embedding(query, model=model)
        query_embedding = normalize([query_embedding], norm='l2')[0]
        query_embedding = np.array([query_embedding]).astype('float32')

        distances, indices = index.search(query_embedding, n)

        results = df.iloc[indices[0]].copy()
        results['distance'] = distances[0]
        return results

    # Answer question based on context
    def answer_question(query, index, df, model=CHAT_MODEL, max_tokens=500):
        top_n = search_embeddings(query, index, df)

        context = "\n\n".join(f"Page {row['page_num']}:\n{row['description']}" for _, row in top_n.iterrows())

        prompt = f"""Answer the following question using the provided context. Cite the page numbers for each fact mentioned in parentheses. If unsure, express uncertainty.

        Context:
        {context}

        Question: {query}

        Answer:"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Ensure to cite page numbers for each fact you provide."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )

        assistant_reply = response.choices[0].message.content
        return assistant_reply

    # Question input
    query = st.text_input("Enter your question about the document:")

    if query:
        st.write("Searching the document and generating a response...")
        answer = answer_question(query, index, df)
        st.write(f"Question: {query}\nAnswer: {answer}")
