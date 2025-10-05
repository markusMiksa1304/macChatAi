import pika
import asyncio
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from core_components import UltimateRAGSetup
from utils import ColorfulStatus
import tomllib

with open("configs.toml", "rb") as f:
    config = tomllib.load(f)

EMBEDDINGS_MODEL = config["general"]["embeddings_model"]
CHROMA_DIR = config["general"]["chroma_dir"]

def callback(ch, method, properties, body):
    task = body.decode()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    rag = UltimateRAGSetup(embeddings, None)
    
    if task.startswith("scrape:"):
        url = task.split(":", 1)[1]
        asyncio.run(rag.scrape_and_index_url(url, "web_scraped"))
    elif task.startswith("scrape_playwright:"):
        url = task.split(":", 1)[1]
        asyncio.run(rag.scrape_with_playwright(url, "playwright_scraped"))
    elif task.startswith("scrape_requests_html:"):
        url = task.split(":", 1)[1]
        asyncio.run(rag.scrape_with_requests_html(url, "requests_html_scraped"))
    elif task.startswith("crawl:"):
        start_url, max_pages = task.split(":", 2)[1:]
        asyncio.run(rag.crawl_and_index_site(start_url, int(max_pages), "web_crawled"))
    else:
        file_path = task
        asyncio.run(rag.index_file(file_path))
    
    ch.basic_ack(delivery_tag=method.delivery_tag)
    ColorfulStatus.success(f"Processed: {task}")

def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='rag_index', durable=True)
    channel.basic_qos(prefetch_count=1)
    
    channel.basic_consume(queue='rag_index', on_message_callback=callback)
    print("Worker gestartet. Warte auf Tasks (Ctrl+C zum Stoppen)...")
    channel.start_consuming()

if __name__ == "__main__":
    main()