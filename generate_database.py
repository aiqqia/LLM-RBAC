from langchain_openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

import openai 
import shutil
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_org_id = os.getenv("OPENAI_ORGANIZATION_ID")

DATA_PATH = "data/textfiles"
CHROMA_PATH = "chroma"

openai_embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key,
    openai_organization=openai_org_id
)


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function=len,
        add_start_index = True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)
    return chunks

def save_to_chroma(chunks: list[Document]):
    #Clear old db first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, openai_embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    
if __name__ == "__main__":
    generate_data_store()