import argparse
import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from pinecone import Pinecone
import openai
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv(find_dotenv())

openai.api_key = os.getenv('openai_api_key')

# Initialize Pinecone
pc = Pinecone(os.getenv('pinecone_api_key'))
index = pc.Index("rag-system")

def process_question_to_embedding(question):
    from llama_index.embeddings.openai import OpenAIEmbedding
    embed_model = OpenAIEmbedding()
    vector = embed_model.get_text_embedding(question)
    return vector


def get_answer(question):
    
    # query_vector = process_question_to_embedding(question)
    # # Include metadata in the results
    # query_result = index.query(vector=query_vector, top_k=5, include_metadata=True)

    # _nodes = []
    # for i, _t in enumerate(query_result['matches']):
    #     try:
    #         _node = TextNode(text=_t['metadata']['text'])
    #         _nodes.append(_node)
    #     except Exception as e:
    #         print(e)

    # # create vector store index
    # _index = VectorStoreIndex(_nodes)
    # llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    # # Re-rank
    # query_engine = _index.as_query_engine(similarity_top_k=5, llm=llm)
    # response = query_engine.query(question)
    # return (str(response))
    query_vector = process_question_to_embedding(question)
    # Query Pinecone index for top results
    query_result = index.query(vector=query_vector, top_k=5, include_metadata=True)

    _nodes = []
    for match in query_result['matches']:
        _nodes.append(TextNode(text=match['metadata']['text']))

    # create vector store index
    _index = VectorStoreIndex(_nodes)
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    # Re-rank
    query_engine = _index.as_query_engine(similarity_top_k=5, llm=llm)
    response = query_engine.query(question)

    # Adding this to return the re-ranked response and a selection of other top responses for comparison
    #other_top_answers = [node.text for node in _nodes[:3]]  # Return the text of top 3 nodes for comparison
    return str(response)







    # # Adding this to return the re-ranked response and a selection of other top responses for comparison
    # other_top_answers = [node.text for node in _nodes[:3]]  # Return the text of top 3 nodes for comparison
    # #return [str(response)] + other_top_answers
    # return str(response)



# def generate_answer_with_gpt(texts):
#     """Generates an answer by combining the context from top texts and querying OpenAI's GPT model."""
#     combined_context = ' '.join(texts)  # Combine the texts into one continuous block of text

#     # Ensure your API key is securely configured and loaded
#     try:
#         # Send a completion request to the OpenAI API
#         response = openai.Completion.create(
#             engine="gpt-3.5-turbo",  # Consider using the latest available model
#             prompt=combined_context,
#             max_tokens=150,
#             temperature=0.7,
#             top_p=0.9,
#             n=1,
#             stop=None  # Specify any stopping criteria, if needed
#         )
#         # Extract the first (and only) response
#         answer = response.choices[0].text.strip()
#         return answer
#     except Exception as e:
#         return f"An error occurred: {str(e)}"



def main():
    parser = argparse.ArgumentParser(description="Query and retrieve answers from Pinecone index.")
    parser.add_argument("--question", type=str, required=False, help="Question to query the indexed data.")
    args = parser.parse_args()
    get_answer(args.question)

if __name__ == "__main__":
    main()

