import argparse
import os
from pinecone import Pinecone 
from sentence_transformers import SentenceTransformer
import openai
os.environ["TOKENIZERS_PARALLELISM"] = "false"



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

# def index_data(texts):
#     for id, text in enumerate(texts):
#         embedding = model.encode(text)
#         # Store both the embedding and the text as metadata
#         index.upsert(items=[(str(id), embedding.tolist(), {'answer': text})])


def query_pinecone(question):
    """
    Queries Pinecone with the provided question and returns the most relevant answers.
    """
    query_vector = process_question_to_embedding(question)
    # Include metadata in the results
    query_results = index.query(vector=query_vector, top_k=5, include_metadata=True)
    texts = [match['metadata']['text'] for match in query_results['matches']]  # Gather texts from results
    return texts


def generate_answer_with_gpt(texts):
    """Generates an answer by combining the context from top texts and querying OpenAI's GPT model."""
    combined_context = ' '.join(texts)  # Combine the texts into one continuous block of text

    # Ensure your API key is securely configured and loaded
    # openai.api_key = ""

    try:
        # Send a completion request to the OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-003",  # Consider using the latest available model
            prompt=combined_context,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            n=1,
            stop=None  # Specify any stopping criteria, if needed
        )
        # Extract the first (and only) response
        answer = response.choices[0].text.strip()
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}"



def main():
    parser = argparse.ArgumentParser(description="Query and retrieve answers from Pinecone index.")
    parser.add_argument("--question", type=str, required=True, help="Question to query the indexed data.")
    args = parser.parse_args()

    # Perform query
    texts = query_pinecone(args.question)
    if texts:
        # Generate an answer using GPT
        answer = generate_answer_with_gpt(texts)
        print("Generated Answer:")
        print(answer)
    else:
        print("No relevant data found to generate an answer.")

if __name__ == "__main__":
    main()

