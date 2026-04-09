import numpy as np
import faiss
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentIndex:
    """Self-contained, immutable index for a single document.

    Returned by Chunker.build_index(). Safe to pass between concurrent
    requests — all state lives here, not on the Chunker instance.
    """
    chunks: list[str]
    embeddings: np.ndarray
    faiss_index: faiss.IndexFlatIP


class Chunker:
    """Splits documents and performs similarity-based retrieval.

    Stateless: no mutable instance state after __init__. Each call to
    build_index() returns an independent DocumentIndex. Concurrent
    requests can safely hold different DocumentIndex objects.
    """

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

    def build_index(self, text: str) -> DocumentIndex:
        """Split text into chunks, embed them, and return a FAISS-backed index."""
        chunks = self.splitter.split_text(text)
        embeddings = self.embedding_model.encode(chunks, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return DocumentIndex(chunks=chunks, embeddings=embeddings, faiss_index=index)

    def retrieve(self, query: str, doc_index: DocumentIndex, top_k: int | None = None) -> list[str]:
        """Retrieve the top-k most similar chunks for a query."""
        k = min(top_k or self.top_k, len(doc_index.chunks))
        query_vec = self.embedding_model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)
        _, indices = doc_index.faiss_index.search(query_vec, k)
        return [doc_index.chunks[i] for i in indices[0] if i < len(doc_index.chunks)]
