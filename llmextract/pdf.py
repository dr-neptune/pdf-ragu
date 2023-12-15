from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from pprint import pprint

from dotenv import load_dotenv
load_dotenv()


def generate_faiss_idx(pdf_location: str) -> FAISS:
    pages = PyPDFLoader(pdf_location).load_and_split()
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    return faiss_index


if __name__ == '__main__':
    faiss_index = generate_faiss_idx('pdfs/apple.pdf')

    results = faiss_index.similarity_search('What was the Cost of sales for year 2018?', k=3)

    for doc in results:
        print(str(doc.metadata['page']) + ':', doc.page_content[:300])
