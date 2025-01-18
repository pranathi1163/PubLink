# PUBMED
A simple and interactive tool for querying, summarizing, and visualizing scientific documents.

## Overview
This project is a streamlined system designed to help users interact with a subset of the PubMed dataset. By selecting a model and inputting a query, the system retrieves the top 3 most relevant documents with similarity scores. Users can then summarize the retrieved documents using an API-powered GROQ model and visualize the summary with a word cloud.

## Features
- **Model Selection:** Choose from various models like TF-IDF, BM25, W2V, Sentence Transformer, or RAG.
- **Query and Retrieve:** Retrieve the top 3 similar documents from the PubMed subset along with similarity scores.
- **Summarization:** Generate a concise summary of the retrieved documents using the GROQ model API.
- **Word Cloud:** Create a visually appealing word cloud based on the summary for quick insights.

## How It Works
### Retrieve:
1. Select a model from the dropdown menu.
2. Enter your query and click **Retrieve**.
3. View the top 3 documents along with similarity scores.

### Summarize:
1. Click the **Summarize** button to generate a summary of the retrieved documents.

### Generate Word Cloud:
1. Click the **Generate Word Cloud** button to create a visual representation of the summary.

### Required Libraries
All necessary dependencies are listed in requirements.txt. Use the installation command mentioned above to set up the environment.

### Usage
1. Launch the application using Streamlit.
2. Enter a query, retrieve documents, summarize them, and generate a word cloud.
3. Interact with the retrieved results and gain insights from the word cloud visualization.