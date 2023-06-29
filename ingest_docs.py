import argparse
import logging

from bot_utils import ingest_from_directory, load_embedding_model
from constants import REDIS_INDEX, REDIS_HOST, EMBEDDING_MODEL

logging.basicConfig(filename='./app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义脚本运行参数，示例python ingest_docs.py --doc_path './docs/github' --redis_host '10.200.180.36:6379'
parser = argparse.ArgumentParser(
    description="ingest raw docs to vectorstore.",
    prog="ingest data"
    )
parser.add_argument("--doc_path", type=str, required=True)
parser.add_argument("--redis_index", type=str, required=False, default=REDIS_INDEX)
parser.add_argument("--redis_host", type=str, required=False, default=REDIS_HOST)
parser.add_argument("--glob", type=str, required=False, default = "*.md")
args = parser.parse_args()

docs_dir = args.doc_path
redis_index = args.redis_index
redis_host = args.redis_host
glob = args.glob

#docs_dir = './raw_docs/github'
embeddings = load_embedding_model(EMBEDDING_MODEL)
#redis_host = "10.200.180.36:6379"

logging.info(f"Docs under {docs_dir} with format {glob} ingesting to index {redis_index}...")
response = ingest_from_directory(
    directory_path=docs_dir,
    glob=glob,
    embeddings=embeddings,
    redis_index=redis_index,
    redis_host=redis_host
    )
print(response)