import streamlit as st
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

# Flask backend URL
backend_url = "http://127.0.0.1:5002"

st.title("Model Selector with Summarization")

# Model selection dropdown
model = st.selectbox("Select Model", ["TF-IDF", "BM25", "W2V", "Sentence Transformer", "RAG"])

# Query input
query = st.text_input("Enter your query:")

# Retrieve Results
if st.button("Retrieve"):
    try:
        response = requests.post(f"{backend_url}/retrieve", json={"query": query, "model": model})
        if response.status_code == 200:
            results = response.json().get("result", [])
            if results:
                result_df = pd.DataFrame(results)
                st.session_state["results_df"] = result_df  # Save results in session state
                st.session_state["abstracts"] = result_df["abstract"].tolist()  # Save abstracts for summarization
                st.success("Results retrieved successfully.")
            else:
                st.warning("No results found.")
        else:
            st.error(response.json().get("error", "Error in retrieval."))
    except Exception as e:
        st.error(f"Error in retrieve: {str(e)}")

# Show retrieved results if they exist
if "results_df" in st.session_state:
    st.write("### Retrieved Results:")
    st.dataframe(st.session_state["results_df"], use_container_width=True)

# Summarize Results
if "abstracts" in st.session_state:
    if st.button("Summarize"):
        try:
            abstracts = st.session_state["abstracts"]
            response = requests.post(f"{backend_url}/summarize", json={"abstracts": abstracts})
            if response.status_code == 200:
                summary = response.json().get("summary", "No summary provided.")
                st.session_state["summary"] = summary  # Save summary in session state
                st.success("Summary generated successfully.")
            else:
                st.error(response.json().get("error", "Error in summarization."))
        except Exception as e:
            st.error(f"Error in summarize: {str(e)}")

# Show summary if it exists
if "summary" in st.session_state:
    st.write("### Summary:")
    st.write(st.session_state["summary"])

# Generate Word Cloud
if "summary" in st.session_state:
    if st.button("Generate Word Cloud"):
        try:
            summary = st.session_state["summary"]
            response = requests.post(f"{backend_url}/wordcloud", json={"summary": summary})
            if response.status_code == 200:
                image_bytes = BytesIO(response.content)
                image = Image.open(image_bytes)
                st.write("### Word Cloud:")
                st.image(image, use_column_width=True)
            else:
                st.error(response.json().get("error", "Error in generating word cloud."))
        except Exception as e:
            st.error(f"Error in word cloud generation: {str(e)}")
