from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer  # Correct import from sklearn
from sklearn.metrics.pairwise import cosine_similarity  # Correct import from sklearn
from transformers import pipeline
import pandas as pd
import os
from groq import Groq
app = Flask(__name__)

# Replace with your actual Groq API Key
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


# Sample models for demonstration
df = pd.read_csv("D:/Packages/sem10/pubmed_archive/covid_abstracts.csv")
df_fillna = df.fillna('unknown')
df_fillna['split_column'] = df_fillna['abstract'].str.split()
df_add_cols = df_fillna
df_add_cols['ID'] = df_add_cols['url'].str.split('/').str[-1]
df_add_cols["title_and_abstract"] = df_add_cols["title"] + " " + df_add_cols["abstract"]

# Initialize summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/')
def home():
    return "Flask backend is running."

@app.route('/retrieve', methods=['POST'])
def retrieve():
    query = request.json.get('query')
    model_name = request.json.get('model')

    if not query or not model_name:
        return jsonify({"error": "'query' and 'model' are required fields."}), 400

    if model_name == "TF-IDF":
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity  # Correct import from sklearn
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df_add_cols["title_and_abstract"])
        query_tfidf = vectorizer.transform([query])
        try:
            # Ensure the input to cosine_similarity is in the correct format (sparse matrix or numpy array)
            similarities = cosine_similarity(query_tfidf, tfidf_matrix)
        except Exception as e:
            return jsonify({"error": f"Cosine similarity error: {str(e)}"}), 500
        # Ensure cosine_similarity is imported from sklearn.metrics.pairwise
        similarities = cosine_similarity(query_tfidf, tfidf_matrix)
        # print(similarities)  # Should print similarity values
        df_add_cols["score"] = similarities.flatten()
        # Sort and get the top 3 results
        result = df_add_cols.sort_values(by="score", ascending=False).head(3)
        result_data = result[["ID", "title", "abstract", "url", "score"]].to_dict(orient="records")

    elif model_name == 'BM25':
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [doc.split() for doc in (df_add_cols["title_and_abstract"])]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        df_add_cols["score"] = bm25_scores
        result = df_add_cols.sort_values(by="score", ascending=False).head(3)

    elif model_name == 'W2V':
        import pandas as pd
        import numpy as np
        from gensim.models import Word2Vec
        from gensim.utils import simple_preprocess
        from sklearn.metrics.pairwise import cosine_similarity
        df_w2v = df_add_cols
        # Tokenize the text
        df_w2v["tokens"] = df_w2v["title_and_abstract"].apply(lambda x: simple_preprocess(x))
        model = Word2Vec.load("D:/Packages/sem10/models/pubmed_word2vec.model")
        # Compute document embeddings - To calculate embeddings for documents, you can average the word embeddings for each token in the document
        def document_embedding(tokens):
            embeddings = [model.wv[word] for word in tokens if word in model.wv]
            if embeddings:
                return np.mean(embeddings, axis=0)
            else:
                return np.zeros(model.vector_size)

        # Add document embeddings to the DataFrame
        df_w2v["doc_embedding"] = df_w2v["tokens"].apply(document_embedding)

        # Tokenize query and calculate embedding
        query_tokens = simple_preprocess(query)
        query_embedding = document_embedding(query_tokens)

        # Calculate cosine similarity
        doc_embeddings = np.array(df_w2v["doc_embedding"].tolist())
        similarities = cosine_similarity([query_embedding], doc_embeddings)

        # Add similarities to the DataFrame
        df_w2v["score"] = similarities.flatten()

        # Retrieve top documents
        result = df_w2v.sort_values(by="score", ascending=False).head(3)

    elif model_name == 'Sentence Transformer':
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        model = SentenceTransformer("D:/Packages/sem10/models/sentence_transformer_saved_model")
        embeddings = model.encode(df_add_cols['title_and_abstract'])
        query_embedding = model.encode(query)
        similarities = cosine_similarity([query_embedding], embeddings)
        df_add_cols["score"] = similarities.flatten()
        result = df_add_cols.sort_values(by="score", ascending=False).head(3)
    
    elif model_name == 'RAG':
        print("rag choosen done")
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
        import pandas as pd
        model = SentenceTransformer("D:/Packages/sem10/models/sentence_transformer_saved_model")
        print("model done")
        embeddings = model.encode(df_add_cols['title_and_abstract'])
        print("encoding done")
        # Initialize FAISS index
        dimension = embeddings.shape[1]  # Embedding size
        index = faiss.IndexFlatL2(dimension)  # L2 distance (use IndexFlatIP for cosine similarity)
        index.add(embeddings)
        print("index done")
        def retrieve(query, k=3):
            print("function")
            query_embedding = model.encode([query], convert_to_numpy=True)
            distances, indices = index.search(query_embedding, k)  # Retrieve top-k results
            results = []
            for i, idx in enumerate(indices[0]):
                print("Index being accessed:", idx)
                print("Available DataFrame indices:", df_add_cols.index)

                results.append({
                    "ID": df_add_cols.iloc[idx]["ID"],
                    "title": df_add_cols.iloc[idx]["title"],
                    "abstract": df_add_cols.iloc[idx]["abstract"],
                    "url": df_add_cols.iloc[idx]["url"],
                    "score": distances[0][i]
                })
            print("fn done")
            return results
        print("call fn done")
        result = retrieve(query, k=3)
        print("fn res done")
        print("RES", result)
        print("Type of res : ", type(result))
        res = pd.DataFrame(result)
        result = res.sort_values(by="score", ascending=False).head(3)

        result_data = result[["ID", "title", "abstract", "url", "score"]].to_dict(orient="records")
        return jsonify({"result": result_data})
        
    else:
        return jsonify({"error": f"Model '{model_name}' is not implemented."}), 400
    result_data = result[["ID", "title", "abstract", "url", "score"]].to_dict(orient="records")
    return jsonify({"result": result_data})


@app.route('/summarize', methods=['POST'])
def summarize():
    client = Groq(api_key=GROQ_API_KEY)
    try:
        data = request.json
        abstracts = data.get("abstracts")
        if not abstracts or not isinstance(abstracts, list):
            return jsonify({"error": "'abstracts' (list of strings) is required for summarization."}), 400
        
        context = " ".join(abstracts)
        max_tokens = 8192
        if len(context.split()) > max_tokens:
            context = " ".join(context.split()[:max_tokens])

        model = data.get("model", "llama3-70b-8192")
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": context}],
            model=model
        )
        print("Groq Response:", chat_completion)  
        # Extract response safely
        if not chat_completion.choices or not chat_completion.choices[0].message:
            return jsonify({"error": "No summary provided by the Groq API."}), 500

        response_content = chat_completion.choices[0].message.content
        return jsonify({"summary": response_content})

    except Exception as e:
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500


@app.route('/wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        from flask import Flask, request, send_file
        from wordcloud import WordCloud
        from io import BytesIO

        data = request.json
        summary = data.get("summary")

        if not summary or not isinstance(summary, str):
            return jsonify({"error": "'summary' (string) is required to generate a word cloud."}), 400

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(summary)

        # Save the word cloud image to a file
        image_path = "wordcloud.png"
        wordcloud.to_file(image_path)

        return send_file(image_path, mimetype='image/png', as_attachment=True)

    except Exception as e:
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500
    

if __name__ == '__main__':
    app.run(port=5002)
