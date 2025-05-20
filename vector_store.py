from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document

class VectorStoreManager:
    def __init__(self, embedding_model):
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        
    def create_vector_store(self, docs):
        self.vector_store = Chroma.from_documents(docs, embedding=self.embedding)
        return self.vector_store
        
    def update_vector_store(self, link):
        docs = self._get_docs_from_web(link)
        if self.vector_store:
            self.vector_store.add_documents(docs)
        else:
            self.create_vector_store(docs)
        return self.vector_store
        
    def _get_docs_from_web(self, link):
        loader = WebBaseLoader(link)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20
        )
        return text_splitter.split_documents(docs)