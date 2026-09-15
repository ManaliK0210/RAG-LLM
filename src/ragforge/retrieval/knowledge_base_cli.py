import argparse
from pathlib import Path

from ragforge.embeddings.config import (
    EmbeddingConfig,
)
from ragforge.embeddings.model import (
    SentenceTransformerEmbedding,
)
from ragforge.retrieval.indexer import (
    VectorIndexer,
)
from ragforge.retrieval.knowledge_base import (
    KnowledgeBase,
)
from ragforge.retrieval.numpy_store import (
    NumpyVectorStore,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vector_store.pkl"
)


def build_knowledge_base(
    source_directory: Path,
    output_path: Path,
) -> int:
    """
    Build the persistent application knowledge base.
    """

    embedding_model = (
        SentenceTransformerEmbedding(
            EmbeddingConfig(
                device="cpu",
            )
        )
    )

    vector_store = NumpyVectorStore(
        dimension=embedding_model.dimension
    )

    indexer = VectorIndexer(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    knowledge_base = KnowledgeBase(
        indexer=indexer,
        vector_store=vector_store,
    )

    count = knowledge_base.build(
        source_directory=source_directory,
        persist_path=output_path,
    )

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the RAGForge persistent "
            "vector knowledge base."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=(
            "Directory containing source documents."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Path for the persisted vector store."
        ),
    )

    args = parser.parse_args()

    count = build_knowledge_base(
        source_directory=args.source,
        output_path=args.output,
    )

    print(
        f"Indexed {count} document chunks."
    )

    print(
        f"Vector store saved to: {args.output}"
    )


if __name__ == "__main__":
    main()