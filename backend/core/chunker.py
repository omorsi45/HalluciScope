import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(
        self,
        embedding_model,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ):
        self.embedding_model = embedding_model
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.top_k = top_k
        self.chunks: list[str] = []
        self.index: faiss.IndexFlatIP | None = None
        self.chunk_embeddings: np.ndarray | None = None

    def index_document(self, text: str) -> None:
        """Split text into chunks and build a FAISS index."""
        self.chunks = self.splitter.split_text(text)
        embeddings = self.embedding_model.encode(self.chunks, normalize_embeddings=True)
        self.chunk_embeddings = np.array(embeddings, dtype=np.float32)
        dim = self.chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.chunk_embeddings)

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        """Retrieve the top-k most similar chunks for a query."""
        if self.index is None:
            return []
        k = min(top_k or self.top_k, len(self.chunks))
        query_vec = self.embedding_model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)
        _, indices = self.index.search(query_vec, k)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
