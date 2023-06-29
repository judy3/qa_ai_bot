import time
import logging
from typing import Any, Optional, Union
from git import Repo
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.document_loaders import JSONLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import RWKV
from langchain.vectorstores.redis import Redis

from constants import (
    EMBEDDING_MODEL,
    RAW_DOC_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTORSEARCH_TOP_K,
    REDIS_HOST,
    REDIS_AUTH_USER,
    REDIS_AUTH_PASS,
    REDIS_INDEX,
    RWKV_MODLE_PATH,
    RWKV_TOKENIZER_PATH,
    RWKV_STRATEGY,
    RESPONSE_MAX_TOKEN,
    RESPONSE_TEMPARATURES,
    JSON_LOADER_KWARGS
)

logging.basicConfig(filename='./app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clone_from_github(
    repo_url: str,
    branch: str,
    user_name: str,
    git_token: str,
    destination_path: Optional[str] = RAW_DOC_PATH,
    depth: Optional[int] = 1, 
    ):
    if '://' in repo_url:
        repo_url_with_creds = f"{repo_url.split('://')[0]}://{user_name}:{git_token}@{repo_url.split('://')[-1]}"
    else:
        repo_url_with_creds = f"https://{user_name}:{git_token}@{repo_url}"
    destination_path = f"{destination_path}/{repo_url.split('/')[-1]}"
    #print(repo_url_with_creds)
    logging.info(f"Cloning {repo_url} to {destination_path}")
    time1 = time.time()
    repo = Repo.clone_from(
        repo_url_with_creds,
        destination_path,
        depth=depth
        )
    repo.git.checkout(branch)
    time2 = time.time()
    logging.info(f"Clone git repository finished")
    return f"Cloned repo {repo_url} on {branch} in {time2-time1} seconds."

def load_embedding_model(
    embedding_model_name: Optional[str] = EMBEDDING_MODEL,
    ):
    logging.info("loading the embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    #print(type(embeddings))
    return embeddings

def load_rwkv_model(
    model_path: Optional[str] = RWKV_MODLE_PATH,
    tokenizer_path: Optional[str] = RWKV_TOKENIZER_PATH,
    strategy: Optional[str] = RWKV_STRATEGY,
    max_token: Optional[int] = RESPONSE_MAX_TOKEN,
    model_temperatures: Optional[float] = RESPONSE_TEMPARATURES
    ):
    logging.info("loading the rwkv model...")
    model = RWKV(model=model_path, strategy=strategy, tokens_path=tokenizer_path)
    model.max_tokens_per_generation = max_token
    model.temperature= model_temperatures
    return model

def ingest_from_directory(
    directory_path: str,
    glob: str,
    embeddings: object,
    loader_class: Optional[Any] = UnstructuredFileLoader,
    use_multithreading: Optional[bool] = True,
    show_progress: Optional[bool] = True,
    chunk_size: Optional[int] = CHUNK_SIZE,
    chunk_overlap: Optional[int] = CHUNK_OVERLAP,
    redis_host: Optional[str] = REDIS_HOST,
    redis_user: Optional[str] = REDIS_AUTH_USER,
    redis_pass: Optional[str] = REDIS_AUTH_PASS,
    redis_index: Optional[str] = REDIS_INDEX,
    loader_kwargs: Union[dict, None] = None
    ):
    # Load files from directory and create chunks of thoes files
    # By default, unstructured files supports texts, md, powerpoints, html, pdfs, images, and more.
    logging.info("Loading the documents...")
    if glob[-5:] == '.json':
        loader_class = JSONLoader
        use_multithreading = False
        if not loader_kwargs: 
            loader_kwargs = JSON_LOADER_KWARGS
    elif glob[-5:] == '.xlsx' or glob[-4:] == '.xls':
        loader_class = UnstructuredExcelLoader
        use_multithreading = False
        #TODO: ERROR needs to be fixed, xlsx files are not supported yet
        #ERROR: ValueError: Excel file format cannot be determined, you must specify an engine manually.
    logging.info(f"document lodaer using is: {loader_class}")
    loader = DirectoryLoader(
        directory_path,
        glob=glob, 
        use_multithreading=use_multithreading,
        show_progress=show_progress,
        loader_cls=loader_class,
        loader_kwargs=loader_kwargs
        )
    raw_docs = loader.load()
    logging.info(f"raw_docs_count: {len(raw_docs)}")
    doc_splitter = NLTKTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
            )
    docs = doc_splitter.split_documents(raw_docs)
    # create vectorstore for the ingested documents and save them locally
    logging.info("Embedding the documents...")
    redis_url = f"redis://{redis_user}:{redis_pass}@{redis_host}"
    logging.info(f"saving docs to redis {redis_host}...")
    vectorstore = Redis.from_documents(
        docs, embeddings, redis_url=redis_url, index_name=redis_index
        )
    return f"The documents are stored in the vectorstore index {redis_index} successfully."

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. If cannot get the answer from the input, just say 'The question cannot be answered based on known information.'

# Instruction:
{instruction}

# Input:
{input}

# Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{instruction}

# Response:
"""

def ask_bot(
    question: str,
    embeddings: object,
    rwkv_model: object,
    redis_host: Optional[str] = REDIS_HOST,
    redis_user: Optional[str] = REDIS_AUTH_USER,
    redis_pass: Optional[str] = REDIS_AUTH_PASS,
    redis_index: Optional[int] = REDIS_INDEX,
    top_k: Optional[int] = VECTORSEARCH_TOP_K
    ):
    logging.info("similarity search in vectorstore...")
    redis_url = f"redis://{redis_user}:{redis_pass}@{redis_host}"
    vectorstore = Redis(
        redis_url=redis_url,
        index_name=redis_index,
        embedding_function=embeddings.embed_query,
        )
    related_docs = vectorstore.similarity_search(
        question,
        k=top_k
    )
    #logging.info(f"searched docs: {related_docs}")
    logging.info("Generating prompt based on the searched results...")
    context = "\n".join([doc.page_content for doc in related_docs])
    logging.info("send searched results to rwkv model and generating the result...")
    prompt = generate_prompt(input=context, instruction=question)
    logging.info(f"The prompt is: {prompt}")
    response = rwkv_model(prompt)
    logging.info(response)

    return response

def ask_bot_without_kb(
    question: str,
    rwkv_model: object
    ):
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
 
# Instruction:
{question}
 
# Response:
    """
    response = rwkv_model(prompt)
    return response



