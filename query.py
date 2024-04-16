import argparse
import os
from pinecone import Pinecone 
from sentence_transformers import SentenceTransformer


# Initialize Pinecone
pc = Pinecone(api_key="93d07fdf-8263-4af9-99e7-1d0a98a4e504")
index = pc.Index("rag-system")

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def process_question_to_embedding(question):
    """
    Converts a question to an embedding using a pre-trained transformer model.
    """
    # Generate the embedding
    embedding = model.encode(question)
    return embedding.tolist()  # Convert numpy array to list if necessary for Pinecone compatibility

def index_data(texts):
    for id, text in enumerate(texts):
        embedding = model.encode(text)
        # Store both the embedding and the text as metadata
        index.upsert(items=[(str(id), embedding.tolist(), {'answer': text})])


def query_pinecone(question):
    """
    Queries Pinecone with the provided question and returns the most relevant answers.
    """
    query_vector = process_question_to_embedding(question)
    # Include metadata in the results
    query_results = index.query(vector=query_vector, top_k=1, include_metadata=True)
    return query_results



def main():
    parser = argparse.ArgumentParser(description="Query and retrieve answers from Pinecone index.")
    parser.add_argument("--question", type=str, required=True, help="Question to query the indexed data.")
    args = parser.parse_args()

    # Perform query
    results = query_pinecone(args.question)
    print("Query Results:")
    for result in results['matches']:
        # Retrieve and print the answer from metadata
        answer = result['metadata']['text']
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
