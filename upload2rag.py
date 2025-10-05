import asyncio
from pathlib import Path
import pika
import tomllib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils import ColorfulStatus

with open("configs.toml", "rb") as f:
    config = tomllib.load(f)

UPLOAD_DIR = Path(config["general"]["upload_docs_dir"])
USE_RABBITMQ = config["rag"]["use_rabbitmq"]

class UploadHandler(FileSystemEventHandler):
    def __init__(self, queue_name="rag_index"):
        self.queue_name = queue_name
        self.connection = None
        self.channel = None
        if USE_RABBITMQ:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.queue_name, durable=True)
    
    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            if Path(file_path).suffix in ['.pdf', '.py', '.md', '.json', '.toml']:
                if USE_RABBITMQ:
                    self.channel.basic_publish(exchange='', routing_key=self.queue_name, body=file_path.encode())
                    ColorfulStatus.info(f"Pushed to RabbitMQ: {file_path}")
                else:
                    asyncio.run(self.index_file(file_path))
    
    def on_modified(self, event):
        self.on_created(event)
    
    async def index_file(self, file_path):
        from core_components import UltimateRAGSetup
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=config["general"]["embeddings_model"])
        rag = UltimateRAGSetup(embeddings, None)
        await rag.index_file(file_path)
    
    def close(self):
        if self.connection:
            self.connection.close()

async def main():
    observer = Observer()
    handler = UploadHandler()
    observer.schedule(handler, str(UPLOAD_DIR), recursive=True)
    observer.start()
    ColorfulStatus.info(f"Watchdog läuft auf {UPLOAD_DIR}...")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        handler.close()
        observer.join()
        ColorfulStatus.success("Indexer gestoppt.")

if __name__ == "__main__":
    asyncio.run(main())