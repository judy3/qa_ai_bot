# default configurations for app in docker containers
#EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_MODEL = './models/all-MiniLM-L6-v2'
RAW_DOC_PATH = './docs'
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 20
VECTORSEARCH_TOP_K = 1
REDIS_HOST = "redis-stack:6379"
REDIS_AUTH_PASS = "II6DYHosuDiJhAac"
REDIS_AUTH_USER = "default"
REDIS_INDEX = "itt_docs" #replace the index name as your want
RWKV_MODLE_PATH = './models/rwkv/RWKV-4-Raven-3B-v12-Eng49%-Chn49%-Jpn1%-Other1%-20230527-ctx4096.pth' #replace with the actual model path
RWKV_TOKENIZER_PATH = './models/rwkv/20B_tokenizer.json'
RWKV_STRATEGY = "cuda fp16" #replace the rwkv strategy as your acutal hardware
RESPONSE_MAX_TOKEN = 1024
RESPONSE_TEMPARATURES = 0.2
# bot api settings
BOT_API_URL = 'http://127.0.0.1:8080/ask_bot'
BOT_API_URL_WITHOUT_KB = 'http://127.0.0.1:8080/ask_bot_without_kb'
# ingest raw docs settings
JSON_LOADER_KWARGS = {
    "jq_schema" : ".[]", # replace this jq_schema vaule as your acutal .json file
    "text_content": False
    }
