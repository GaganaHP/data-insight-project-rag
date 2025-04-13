import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from rag.pinecone_client import pc, index
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
generator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def retrieve_context(query, top_k=5):
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    chunks = [match['metadata']['text'] for match in results['matches']]
    return chunks

def re_rank_results(query, retrieved_chunks):
    pairs = [[query, chunk] for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
    ranked_chunks = [text for text, score in ranked]
    return ranked_chunks

def generate_answer(query):
    retrieved_chunks = retrieve_context(query)
    ranked_chunks = re_rank_results(query, retrieved_chunks)
    context = "\n".join(ranked_chunks[:3])
    prompt = f"Using the following context:\n{context}\nAnswer the question: {query}\n"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    output = generator_model.generate(**inputs, max_length=256)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# For testing directly
if __name__ == "__main__":
    sample_query = "How can I connect to a database in Python?"
    print("Answer:", generate_answer(sample_query))
