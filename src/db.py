import logging
import chromadb
from chromadb.utils import embedding_functions
from .config import MODEL_NAME, COLLECTION_NAME, VECTORSTORE_PATH

logger = logging.getLogger(__name__)


def get_collection(
    model_name: str = MODEL_NAME,
    collection_name: str = COLLECTION_NAME,
    vectorstore_path: str = VECTORSTORE_PATH,
):
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    client = chromadb.PersistentClient(path=vectorstore_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )
    logger.info("Collection '%s' ready — %d documents", collection.name, collection.count())
    return collection
