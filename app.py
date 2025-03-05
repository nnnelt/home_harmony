import streamlit as st
import faiss
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load IKEA product data
ikea_df = pd.read_csv("ikea_all_products.csv")  # Ensure this file is available

# Load extracted image features
df = pd.read_csv("image_features.csv")
image_paths = df["image_path"].values
image_embeddings = df.iloc[:, 1:].values.astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(image_embeddings.shape[1])
index.add(image_embeddings)

# Function to find similar images
def find_similar_images(query_image, top_k=5):
    # Process query image
    query_image = preprocess(query_image).unsqueeze(0).to(device)
    
    # Generate embedding
    with torch.no_grad():
        query_embedding = model.encode_image(query_image).cpu().numpy()

    # Search in FAISS
    distances, indices = index.search(query_embedding, top_k)
    
    # Get matching image paths
    results = []
    for idx in indices[0]:
        image_path = image_paths[idx]
        # Try to find the IKEA product info based on the image filename
        product_info = ikea_df[ikea_df["image_url"].str.contains(image_path.split("/")[-1], na=False, case=False)]
        if not product_info.empty:
            product_details = product_info.iloc[0]
            results.append({
                "image_path": image_path,
                "title": product_details["title"],
                "price": product_details["price"],
                "link": product_details["link"]
            })
        else:
            results.append({"image_path": image_path, "title": "Unknown", "price": "N/A", "link": ""})

    return results

# Streamlit UI
st.title("🛋️ Home Harmony: AI-Powered Furniture Search")

uploaded_file = st.file_uploader("Upload an image to find similar furniture", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Uploaded Image", use_column_width=True)

    # Perform search
    st.write("🔍 Searching for similar furniture...")
    similar_items = find_similar_images(query_image)

    # Display results
    for item in similar_items:
        st.image(item["image_path"], caption=f"{item['title']} - {item['price']}")
        if item["link"]:
            st.markdown(f"[View Product]({item['link']})")