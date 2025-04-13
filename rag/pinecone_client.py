# rag/pinecone_client.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Initialize Pinecone client with ServerlessSpec
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "document-qa-index"
dimension = 384  # same as MiniLM-L6-v2

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Adjust based on your embedding model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Free plan supports only this region
        )
    )

index = pc.Index(index_name)
