from datetime import datetime

MODEL_NAME       = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME  = "mmm_outputs"
VECTORSTORE_PATH = "./mmm_vectorstore"
DATA_FOLDER      = "mmm_dummy_data"
LLM_MODEL        = "gemini/gemini-3.5-flash"
CURRENT_YEAR     = str(datetime.now().year)
LAST_YEAR        = str(datetime.now().year - 1)
