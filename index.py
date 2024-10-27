import os
import json
import hashlib
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Access API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("quickstart")

# Set up embeddings with the Google API key
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def chunk_text_for_list(docs: list[str], max_chunk_size: int = 1000) -> list[list[str]]:
    """Splits text into manageable chunks, skipping empty chunks."""
    def chunk_text(text: str, max_chunk_size: int) -> list[str]:
        if not text.endswith("\n\n"):
            text += "\n\n"
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()  # Strip whitespace
            if not paragraph:  # Skip empty paragraphs
                continue
            if len(current_chunk) + len(paragraph) + 2 > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += paragraph + "\n\n"
        if current_chunk.strip():  # Add last chunk if non-empty
            chunks.append(current_chunk.strip())
        return chunks

    chunked_docs = [chunk_text(doc, max_chunk_size) for doc in docs]
    return chunked_docs

def generate_short_id(content: str) -> str:
    """Generate a short ID based on the content using SHA-256 hash."""
    hash_obj = hashlib.sha256()
    hash_obj.update(content.encode("utf-8"))
    return hash_obj.hexdigest()

def combine_vector_and_text(company: dict, doc_embeddings: list[list[float]], chunks: list[str]) -> list[dict[str, any]]:
    """Combine embeddings with metadata, skipping empty text chunks."""
    data_with_metadata = []
    company_name = company["company_name"]

    for chunk, embedding in zip(chunks, doc_embeddings):
        if not chunk.strip():  # Skip empty text chunks
            continue
        
        doc_id = generate_short_id(chunk)
        data_item = {
            "id": doc_id,
            "values": embedding,
            "metadata": {
                "company_name": company_name,
                "text": chunk
            }
        }
        data_with_metadata.append(data_item)

    return data_with_metadata

def embed_chunked_company_data(json_file_path, max_chunk_size=1000):
    with open(json_file_path, "r") as file:
        company_data = json.load(file)

    all_data_with_metadata = []

    for company in company_data:
        text_to_embed = company["description"]
        
        # Filter out empty or whitespace-only descriptions
        if not text_to_embed.strip():
            continue
        
        # Chunk the text
        chunks = chunk_text_for_list([text_to_embed], max_chunk_size)[0]
        
        # Remove any empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        if not chunks:
            continue  # Skip if no valid chunks remain
        
        # Generate embeddings for each chunk
        doc_embeddings = embeddings.embed_documents(chunks)
        
        # Combine data with metadata for upserting
        data_with_metadata = combine_vector_and_text(company, doc_embeddings, chunks)
        all_data_with_metadata.extend(data_with_metadata)

    # Remove entries with empty 'text' values in metadata
    all_data_with_metadata = [
        item for item in all_data_with_metadata
        if item["metadata"].get("text", "").strip()
    ]

    return all_data_with_metadata

def upsert_data_to_pinecone(data_with_metadata: list[dict[str, any]]) -> None:
    """Upsert data with metadata into a Pinecone index."""
    index.upsert(vectors=data_with_metadata)

# Upsert only if data_with_metadata has valid entries
data_with_meta_data = embed_chunked_company_data("./CompanyData.json")
if data_with_meta_data:
    upsert_data_to_pinecone(data_with_metadata=data_with_meta_data)
else:
    print("No valid data to upsert.")

# Query embedding
def get_query_embeddings(query: str) -> list[float]:
    """This function returns a list of the embeddings for a given query.

    Args:
        query (str): The actual query/question.

    Returns:
        list[float]: The embeddings for the given query.
    """
    query_embeddings = embeddings.embed_query(query)
    return query_embeddings

# Example of calling the function
query_embeddings = get_query_embeddings(query="What services are available in InnoTech?")

def query_pinecone_index(
    query_embeddings: list[float],
    top_k: int = 1,
    include_metadata: bool = True,
    company_id: str = None
) -> dict[str, any]:
    """
    Query a Pinecone index for a specific company using a filter on metadata.

    Args:
    - query_embeddings (list[float]): List of query vectors.
    - top_k (int): Number of nearest neighbors to retrieve (default: 2).
    - include_metadata (bool): Whether to include metadata in the query response.
    - company_id (str): The unique company ID to filter by.

    Returns:
    - query_response (dict[str, any]): Query response containing nearest neighbors.
    """
    # filter_dict = {"company_id": company_id} if company_id else None
    query_response = index.query(
        vector=query_embeddings,
        filter={"company_name": {"$eq": "InnoTech Solutions"}},
        top_k=top_k,
        include_metadata=include_metadata,
    )
    return query_response

# Example of calling the function with a filter
query_response = query_pinecone_index(query_embeddings=query_embeddings, company_id="c54b9f2660088ffe46edfc323baf27817f9caa114c73512d52b591833a26e36a")
if query_response and "matches" in query_response:
    print("Matches received from Pinecone:", query_response['matches'])
else:
    print("No matches found in the Pinecone response.")


# Instantiate the LLM using Gemini
LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Extract only the text from the query response
text_answer = " ".join([doc['metadata']['text'] for doc in query_response['matches']])

# Create the prompt to generate a better answer
prompt = f"{text_answer} Using the provided information, give me a better and summarized answer."

def better_query_response(prompt: str) -> str:
    """This function returns a better response using the LLM.
    
    Args:
        prompt (str): The prompt to send to the LLM.

    Returns:
        str: The actual response returned by the LLM.
    """
    # Format the prompt for the conversation model
    conversation_history = [
        {"role": "system", "content": "You are a helpful assistant that provides summarized answers."},
        {"role": "user", "content": prompt}
    ]

    # Call the LLM with the formatted conversation history
    better_answer = LLM.invoke(conversation_history)
    return better_answer.content  # Return the generated answer


# Call the function to get the final answer
final_answer = better_query_response(prompt=prompt)
print(final_answer)
