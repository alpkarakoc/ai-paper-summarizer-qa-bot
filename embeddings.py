from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

class EmbeddingRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.db = None

    def build_vectorstore(self, raw_text):
        texts = self.text_splitter.split_text(raw_text)
        self.db = FAISS.from_texts(texts, self.embedding_model)

    def retrieve_relevant_context(self, question, k=2):
        if not self.db:
            raise ValueError("Vectorstore not built yet.")
        docs = self.db.similarity_search(question, k=k)
        return " ".join([doc.page_content for doc in docs])
