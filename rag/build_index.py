from dotenv import load_dotenv
load_dotenv()

import os
import glob
# from pinecone import Pinecone, ServerlessSpec
from pinecone_client import pc, index
from sentence_transformers import SentenceTransformer

# # Initialize Pinecone client with ServerlessSpec
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# index_name = "document-qa-index"
# dimension = 384  # same as MiniLM-L6-v2

# # Create index if it doesn't exist
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=dimension,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )

# # Connect to the index
# index = pc.Index(index_name)

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def index_documents(pdf_folder="data/docs/"):
    for pdf_path in glob.glob(pdf_folder + "/*.pdf"):
        with open(pdf_path, "rb") as f:
            full_text = f.read().decode("utf-8", errors="ignore")
        chunks = full_text.split("\n")
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:
                embedding = model.encode(chunk).tolist()
                doc_id = f"{os.path.basename(pdf_path)}_{i}"
                index.upsert(vectors=[(doc_id, embedding, {"text": chunk})])
    print("Indexing complete!")

if __name__ == "__main__":
    index_documents()
