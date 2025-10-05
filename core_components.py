import os
import json
import subprocess
import re
import ast
from pathlib import Path
from typing import Dict, Any
import asyncio
from langchain.memory import (
    ConversationSummaryBufferMemory, EntityMemory, KnowledgeGraphMemory,
    ConversationBufferWindowMemory
)
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    UnstructuredPDFLoader, PyPDFLoader, ArxivLoader, DirectoryLoader, TextLoader, PythonLoader,
    JSONLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pika
import tomllib
import requests
from bs4 import BeautifulSoup
import scrapy
from scrapy.crawler import CrawlerProcess
import feedparser
from playwright.async_api import async_playwright
from requests_html import AsyncHTMLSession
from utils import ColorfulStatus, mcp_api_tool
from datetime import datetime
from rich import print as rprint

with open("configs.toml", "rb") as f:
    config = tomllib.load(f)

CHUNK_SIZE = config["rag"]["chunk_size"]
CHUNK_OVERLAP = config["rag"]["chunk_overlap"]
PDF_MODE = config["rag"]["pdf_mode"]
REPO_ANALYZE = config["rag"]["repo_analyze"]
GRADE_THRESHOLD = config["rag"]["grade_threshold"]
CONTEXT_ISOLATE = config["memory"]["context_isolate"]
REDIS_URL = config["memory"]["redis_url"]
MEMORY_WINDOW = config["general"]["memory_window"]
CHROMA_DIR = config["general"]["chroma_dir"]
UPLOAD_DIR = config["general"]["upload_docs_dir"]
MLX_QUANT = config["mlx_tuning"]["quant"]
MLX_ATTN = config["mlx_tuning"]["attn_mode"]
MLX_LORA = config["mlx_tuning"]["lora_path"]
MLX_BATCH = config["mlx_tuning"]["batch_size"]

class UltimateMemoryManager:
    def __init__(self, topics_dir="./topics", llm=None):
        self.topics_dir = Path(topics_dir)
        self.topics_dir.mkdir(exist_ok=True)
        self.current_topic = config["general"]["current_topic"]
        self.memories: Dict[str, Dict] = {}
        self.llm = llm
        self.current_type = config["memory"]["default_type"]
        self.load_topics()
    
    def set_llm(self, llm):
        self.llm = llm
    
    async def get_memory(self, topic: str, memory_type: str = None):
        if memory_type:
            self.current_type = memory_type
        if topic not in self.memories or self.memories[topic]["type"] != self.current_type:
            history = RedisChatMessageHistory(session_id=topic, url=REDIS_URL) if CONTEXT_ISOLATE and REDIS_URL else None
            if self.current_type == "summary_buffer":
                memory = ConversationSummaryBufferMemory(
                    llm=self.llm, max_token_limit=200, k=MEMORY_WINDOW,
                    return_messages=True, output_key="output_text",
                    chat_memory_key="chat_history", history=history
                )
            elif self.current_type == "entity":
                memory = EntityMemory(
                    llm=self.llm, k=10, return_messages=True, state_key="entities",
                    history=history
                )
            elif self.current_type == "kg":
                memory = KnowledgeGraphMemory(
                    llm=self.llm, return_messages=True, k=2,
                    history=history
                )
            else:
                memory = ConversationBufferWindowMemory(
                    k=MEMORY_WINDOW, return_messages=True,
                    human_prefix="You", ai_prefix="AI", history=history
                )
            self.memories[topic] = {"instance": memory, "type": self.current_type}
        
        self.current_topic = topic
        memory = self.memories[topic]["instance"]
        
        if len(memory.chat_memory.messages) > MEMORY_WINDOW * 2:
            summary_chain = load_summarize_chain(self.llm, chain_type="map_reduce")
            summary = await asyncio.to_thread(summary_chain.arun, memory.chat_memory.messages)
            memory.chat_memory.clear()
            memory.chat_memory.add_ai_message(f"Summary: {summary}")
        
        if CONTEXT_ISOLATE and hasattr(memory, 'chat_memory') and REDIS_URL:
            memory.chat_memory.add_filter(lambda msg: len(msg.content) > 10)
        
        return memory
    
    async def switch_topic(self, topic: str, memory_type: str = None):
        await self.get_memory(topic, memory_type)
        print(f"🎯 Topic '{topic}' mit Memory '{self.current_type}' geladen.")
    
    async def save_topics(self):
        for topic, data in self.memories.items():
            memory = data["instance"]
            export = {
                "type": data["type"],
                "messages": [msg.dict() for msg in memory.chat_memory.messages],
                "entities": getattr(memory, 'entity_store', {}),
                "kg_graph": getattr(memory, 'kg', None)
            }
            with open(self.topics_dir / f"{topic}.json", "w") as f:
                json.dump(export, f, indent=2)
        print("💾 Serialisierung gespeichert!")
    
    async def load_topics(self):
        for topic_file in self.topics_dir.glob("*.json"):
            topic = topic_file.stem
            with open(topic_file) as f:
                data = json.load(f)
                memory = await self.get_memory(topic, data.get("type", "buffer"))
                for msg in data.get("messages", []):
                    if "human" in msg.get("type", ""):
                        memory.chat_memory.add_user_message(msg["content"])
                    else:
                        memory.chat_memory.add_ai_message(msg["content"])
                if "entities" in data:
                    memory.entity_store = data["entities"]
    
    async def memory_status(self):
        memory = self.memories.get(self.current_topic, {}).get("instance")
        if memory:
            print(f"📊 Status für '{self.current_topic}': Type={self.current_type}, Messages={len(memory.chat_memory.messages)}")
            if hasattr(memory, 'entity_store'):
                print(f"Entities: {list(memory.entity_store.keys())}")
            if hasattr(memory, 'kg'):
                print(f"KG-Edges: {len(memory.kg.get('edges', []))}")

    async def clear_memory(self, topic: str = None):
        topic = topic or self.current_topic
        if topic in self.memories:
            del self.memories[topic]
            print(f"🗑️ Memory für '{topic}' gelöscht.")
    
    async def export_memory(self, topic: str = None):
        topic = topic or self.current_topic
        data = {
            "topic": topic,
            "type": self.current_type,
            "messages": [msg.dict() for msg in self.memories[topic]["instance"].chat_memory.messages],
            "entities": getattr(self.memories[topic]["instance"], 'entity_store', {}),
            "kg_graph": getattr(self.memories[topic]["instance"], 'kg', None)
        }
        file = self.topics_dir / f"{topic}_export.json"
        with open(file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"📤 Export: {file}")

class JSONDialogLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
    
    def log_turn(self, dialog_id: str, user_input: str, prefill: str, llm_call: Dict, tools_used: list, memory_state: Dict, rag_docs: list = []):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "prefill": prefill,
            "llm_call": llm_call,
            "tools_used": tools_used,
            "memory_state": memory_state,
            "rag_docs": rag_docs
        }
        log_file = self.log_dir / f"{dialog_id}.json"
        logs = []
        if log_file.exists():
            with open(log_file) as f:
                logs = json.load(f)
        logs.append(log_entry)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)

class MLXOptimizer:
    @staticmethod
    def extract_hf_repo(ollama_name):
        try:
            result = subprocess.run(["ollama", "show", ollama_name, "--modelfile"], capture_output=True, text=True, check=True)
            modelfile = result.stdout
            from_match = re.search(r'FROM\s+(.+)', modelfile.strip())
            template_match = re.search(r'TEMPLATE\s+(.+)', modelfile)
            return from_match.group(1).strip() if from_match else None, template_match.group(1).strip() if template_match else None
        except:
            return None, None
    
    @staticmethod
    def check_hf_version(repo):
        try:
            resp = requests.get(f"https://huggingface.co/api/models/{repo}")
            if resp.status_code == 200:
                print(f"📊 {repo}: Letzter Update {resp.json().get('lastModified', 'unknown')}")
        except:
            pass
    
    @staticmethod
    def convert_to_mlx_optimized(ollama_name, hf_repo, template=None, output_dir="./mlx_models", quant=MLX_QUANT):
        mlx_path = Path(output_dir) / ollama_name
        if mlx_path.exists():
            print(f"✓ Bereits optimiert: {mlx_path}")
            return MLXOptimizer.benchmark_mlx(mlx_path)
        
        Path(output_dir).mkdir(exist_ok=True)
        cmd = [
            "mlx_lm.convert",
            "--hf-path", hf_repo,
            "--mlx-path", str(mlx_path),
            "--quantize", f"dynamic{quant}",
            "--attn-mode", MLX_ATTN,
            "--vt-decoder"
        ]
        if MLX_LORA:
            cmd += ["--lora-path", MLX_LORA]
        if template:
            cmd += ["--chat-template", template]
        
        print(f"🔄 Optimiere {ollama_name} (Dynamic Q{quant}, {MLX_ATTN}, VT)...")
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Optimiert!")
            return MLXOptimizer.benchmark_mlx(mlx_path)
        except:
            print(f"✗ Fallback zu Q8...")
            return MLXOptimizer.convert_to_mlx_optimized(ollama_name, hf_repo, template, output_dir, 8)
    
    @staticmethod
    def benchmark_mlx(mlx_path):
        test_prompt = "Benchmark: Erkläre KI."
        cmd = [
            "mlx_lm.generate", "--model", str(mlx_path),
            "--prompt", test_prompt, "--max-tokens", "50", "--temp", "0.1",
            "--batch-size", str(MLX_BATCH), "--top-p", "0.9", "--top-k", "40"
        ]
        start = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        start.wait()
        output = start.stdout.read().decode()
        tokens_est = len(output.split())
        print(f"⚡ Benchmark: ~{tokens_est / (start.returncode + 1)} Tokens/s")
        return True

class UltimateRAGSetup:
    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm
        self.tools = []
        self.vectorstores = {}
        self.observer = None
        self.prompts_cache = None
    
    def load_external_prompts(self, prompts_dir: Path):
        if self.prompts_cache:
            return self.prompts_cache
        prompts_text = ""
        for md_file in prompts_dir.glob("*.md"):
            with open(md_file, "r") as f:
                prompts_text += f"\n--- {md_file.name} ---\n{f.read()}\n"
        self.prompts_cache = prompts_text
        return prompts_text
    
    async def add_pdf(self, file_path: str, topic: str, browse_pages: str = None):
        file_path = Path(file_path)
        if browse_pages:
            from langchain_community.document_loaders import PDFPlumberLoader
            loader = PDFPlumberLoader(file_path, parse_tables=True)
            docs = loader.load()
            page_ranges = [int(p) for p in browse_pages.split(",") if p.isdigit()]
            filtered_docs = [doc for i, doc in enumerate(docs) if i+1 in page_ranges]
        else:
            loader = UnstructuredPDFLoader(file_path, mode=PDF_MODE, strategy="hi_res")
            docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n### ", "\n## ", "\n# ", "\n"]
        )
        splits = splitter.split_documents(docs)
        vs = await asyncio.to_thread(Chroma.from_documents, splits, self.embeddings, persist_directory=f"{CHROMA_DIR}/{topic}")
        vs.persist()
        self.vectorstores[topic] = vs
        print(f"📄 PDF {file_path} (Browse: {browse_pages}) in {topic} geladen.")
    
    async def search_arxiv(self, query: str, topic: str, num_docs=3):
        from langchain_community.document_loaders import ArxivLoader
        loader = ArxivLoader(query=query, load_max_docs=num_docs)
        docs = await asyncio.to_thread(loader.load)
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = splitter.split_documents(docs)
        vs = await asyncio.to_thread(Chroma.from_documents, splits, self.embeddings, persist_directory=f"{CHROMA_DIR}/{topic}")
        vs.persist()
        self.vectorstores[topic] = vs
        print(f"🔬 ArXiv '{query}' geladen in {topic}.")
    
    async def index_file(self, file_path: str):
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        docs = []
        if ext == ".pdf":
            await self.add_pdf(str(file_path), "global")
            return
        elif ext == ".py":
            from langchain_community.document_loaders import PythonLoader
            loader = PythonLoader(file_path)
            docs = loader.load()
            if REPO_ANALYZE:
                with open(file_path) as f:
                    tree = ast.parse(f.read())
                structure = self.extract_py_structure(tree)
                docs.append(Document(page_content=f"Repo-Struktur ({file_path.name}): {structure}", metadata={"source": file_path}))
        elif ext == ".json":
            from langchain_community.document_loaders import JSONLoader
            loader = JSONLoader(file_path, jq_schema=".[]")
            docs = loader.load()
        elif ext == ".toml":
            with open(file_path) as f:
                data = tomllib.load(f)
            docs.append(Document(page_content=f"TOML-Struktur: {json.dumps(data, indent=2)}", metadata={"source": file_path}))
        elif ext == ".md":
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
        else:
            loader = TextLoader(file_path)
            docs = loader.load()
        
        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            splits = splitter.split_documents(docs)
            vs = await asyncio.to_thread(Chroma.from_documents, splits, self.embeddings, persist_directory=f"{CHROMA_DIR}/global")
            vs.persist()
            print(f"📂 Indiziert: {file_path} ({len(splits)} Chunks)")
    
    async def index_file_from_string(self, content: str, topic: str):
        doc = Document(page_content=content, metadata={"source": "dynamic"})
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = splitter.split_documents([doc])
        vs = await asyncio.to_thread(Chroma.from_documents, splits, self.embeddings, persist_directory=f"{CHROMA_DIR}/{topic}")
        vs.persist()
        print(f"📄 Indiziert String in {topic} ({len(splits)} Chunks)")
    
    async def scrape_and_index_url(self, url: str, topic: str = "scraped"):
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})
            soup = BeautifulSoup(response.content, 'lxml')
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            content = "\n".join([elem.get_text(strip=True) for elem in soup.select('.content, .main, .article, .product')])
            doc = Document(page_content=content or soup.get_text(separator='\n', strip=True), metadata={"source": url, "type": "scraped"})
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            splits = splitter.split_documents([doc])
            vs = await asyncio.to_thread(Chroma.from_documents, splits, self.embeddings, persist_directory=f"{CHROMA_DIR}/{topic}")
            vs.persist()
            print(f"📄 Gescrapt & indiziert: {url} ({len(splits)} Chunks)")
            return content[:200] + "..."
        except Exception as e:
            print(f"✗ Scrap-Fehler {url}: {e}")
            return None
    
    async def scrape_with_playwright(self, url: str, topic: str = "playwright_scraped"):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(1000)
                content = await page.content()
                await browser.close()
            soup = BeautifulSoup(content, 'lxml')
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            content = "\n".join([elem.get_text(strip=True) for elem in soup.select('.content, .main, .article, .product')])
            doc = Document(page_content=content or soup.get_text(separator='\n', strip=True), metadata={"source": url, "type": "playwright"})
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            splits = splitter.split_documents([doc])
            vs = await asyncio.to_thread(Chroma.from_documents, splits, self.embeddings, persist_directory=f"{CHROMA_DIR}/{topic}")
            vs.persist()
            print(f"📄 Playwright gescrapt & indiziert: {url} ({len(splits)} Chunks)")
            return content[:200] + "..."
        except Exception as e:
            print(f"✗ Playwright-Fehler {url}: {e}")
            return None
    
    async def scrape_with_requests_html(self, url: str, topic: str = "requests_html_scraped"):
        try:
            session = AsyncHTMLSession()
            response = await session.get(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
            await response.html.arender(timeout=30, sleep=1)
            soup = BeautifulSoup(response.html.html, 'lxml')
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            content = "\n".join([elem.get_text(strip=True) for elem in soup.select('.content, .main, .article, .product')])
            doc = Document(page_content=content or soup.get_text(separator='\n', strip=True), metadata={"source": url, "type": "requests_html"})
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            splits = splitter.split_documents([doc])
            vs = await asyncio.to_thread(Chroma.from_documents, splits, self.embeddings, persist_directory=f"{CHROMA_DIR}/{topic}")
            vs.persist()
            await session.close()
            print(f"📄 Requests-HTML gescrapt & indiziert: {url} ({len(splits)} Chunks)")
            return content[:200] + "..."
        except Exception as e:
            print(f"✗ Requests-HTML-Fehler {url}: {e}")
            return None
    
    async def call_mcp_api(self, query: str, topic: str = "mcp_api"):
        try:
            content = mcp_api_tool(query)
            if "Fehler" in content:
                print(f"✗ {content}")
                return content
            doc = Document(page_content=content, metadata={"source": query, "type": "mcp_api"})
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            splits = splitter.split_documents([doc])
            vs = await asyncio.to_thread(Chroma.from_documents, splits, self.embeddings, persist_directory=f"{CHROMA_DIR}/{topic}")
            vs.persist()
            print(f"📄 MCP-API aufgerufen & indiziert: {query} ({len(splits)} Chunks)")
            return content[:200] + "..."
        except Exception as e:
            print(f"✗ MCP-API-Fehler: {e}")
            return None
    
    async def fetch_and_index(self, query: str, topic: str = "auto_fetched"):
        from utils import auto_tool_selector
        tool_name, confidence = auto_tool_selector(query, self.llm)
        print(f"📍 Auto-Tool: {tool_name} (Confidence: {confidence:.2f})")
        
        if tool_name == "scrape_url":
            result = await self.scrape_and_index_url(query, topic)
        elif tool_name == "scrape_playwright":
            result = await self.scrape_with_playwright(query, topic)
        elif tool_name == "scrape_requests_html":
            result = await self.scrape_with_requests_html(query, topic)
        elif tool_name == "mcp_api":
            result = await self.call_mcp_api(query, topic)
        elif tool_name == "rss_feed":
            from utils import rss_feed_tool
            content = rss_feed_tool(query)
            doc = Document(page_content=content, metadata={"source": query, "type": "rss_feed"})
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            splits = splitter.split_documents([doc])
            vs = await asyncio.to_thread(Chroma.from_documents, splits, self.embeddings, persist_directory=f"{CHROMA_DIR}/{topic}")
            vs.persist()
            print(f"📄 RSS-Feed gescrapt & indiziert: {query} ({len(splits)} Chunks)")
            result = content[:200] + "..."
        else:
            result = "Unbekanntes Tool"
        
        return result, tool_name, confidence
    
    async def crawl_and_index_site(self, start_url: str, max_pages: int, topic: str = "crawled"):
        class DynamicSpider(scrapy.Spider):
            name = 'dynamic'
            start_urls = [start_url]
            custom_settings = {
                'DEPTH_LIMIT': max_pages,
                'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'DOWNLOAD_DELAY': 1.0
            }
            
            def parse(self, response):
                soup = BeautifulSoup(response.text, 'lxml')
                for tag in soup(['script', 'style', 'nav', 'footer']):
                    tag.decompose()
                content = "\n".join([elem.get_text(strip=True) for elem in soup.select('.content, .main, .article')])
                yield {'content': content or soup.get_text(separator='\n', strip=True), 'url': response.url}
                
                for link in response.css('a::attr(href)').getall():
                    if link.startswith('http'):
                        yield response.follow(link, self.parse)
        
        output_file = Path(CHROMA_DIR) / f"{topic}.json"
        process = CrawlerProcess({
            'FEEDS': {str(output_file): {'format': 'json'}},
            'LOG_LEVEL': 'INFO'
        })
        process.crawl(DynamicSpider)
        process.start()
        
        if output_file.exists():
            with open(output_file) as f:
                data = json.load(f)
            docs = [Document(page_content=item['content'], metadata={"source": item['url'], "type": "crawled"}) for item in data]
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            splits = splitter.split_documents(docs)
            vs = await asyncio.to_thread(Chroma.from_documents, splits, self.embeddings, persist_directory=f"{CHROMA_DIR}/{topic}")
            vs.persist()
            print(f"📄 Gecrawlt & indiziert: {len(docs)} Seiten ({len(splits)} Chunks)")
    
    def extract_py_structure(self, tree):
        structure = {"imports": [], "classes": [], "functions": []}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                structure["imports"].append([alias.name for alias in node.names])
            elif isinstance(node, ast.ClassDef):
                structure["classes"].append({"name": node.name, "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)]})
            elif isinstance(node, ast.FunctionDef):
                structure["functions"].append(node.name)
        return json.dumps(structure, indent=2)
    
    def get_retriever(self, topic: str):
        if topic in self.vectorstores:
            return self.vectorstores[topic].as_retriever(search_kwargs={"k": 3})
        return None
    
    async def parallel_index(self, dir_path: str):
        dir_path = Path(dir_path)
        if config["rag"]["use_rabbitmq"]:
            connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
            channel = connection.channel()
            channel.queue_declare(queue='rag_index', durable=True)
            for file in dir_path.rglob("*"):
                if file.suffix in ['.pdf', '.py', '.md', '.json', '.toml']:
                    channel.basic_publish(exchange='', routing_key='rag_index', body=str(file).encode())
            connection.close()
            print(f"Tasks gepusht zu RabbitMQ: {dir_path}")
        else:
            tasks = [self.index_file(str(file)) for file in dir_path.rglob("*") if file.suffix in ['.pdf', '.py', '.md', '.json', '.toml']]
            await asyncio.gather(*tasks)
        print(f"Parallel Index abgeschlossen: {dir_path}")
    
    async def start_watchdog(self):
        class WatchHandler(FileSystemEventHandler):
            def __init__(self, rag):
                self.rag = rag
            
            async def on_created(self, event):
                if not event.is_directory:
                    await self.rag.index_file(event.src_path)
            
            async def on_modified(self, event):
                if not event.is_directory:
                    await self.rag.index_file(event.src_path)
        
        self.observer = Observer()
        handler = WatchHandler(self)
        self.observer.schedule(handler, str(UPLOAD_DIR), recursive=True)
        self.observer.start()
        print(f"👀 Watchdog läuft auf {UPLOAD_DIR}.")
    
    async def stop_watchdog(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
            print("Watchdog gestoppt.")
    
    async def adaptive_rag(self, query: str, topic: str):
        prompts_dir = Path(config["general"]["prompts_dir"])
        prompts_text = self.load_external_prompts(prompts_dir)
        
        retriever = self.get_retriever(topic)
        if not retriever:
            return "No Docs – Fallback to Web Search"
        
        docs = retriever.get_relevant_documents(query)[:5]
        
        grade_template = PromptTemplate.from_template(
            f"{prompts_text}\nGrade Relevanz: Query: {{query}}\nDoc: {{doc}}\nRelevant? (Y/N + Score 1-10):"
        )
        grade_tasks = [self.llm.ainvoke(grade_template.format(query=query, doc=doc.page_content)) for doc in docs]
        grades = await asyncio.gather(*grade_tasks)
        
        relevant_docs = []
        scores = []
        for i, grade in enumerate(grades):
            score_str = re.search(r'\d+', grade.content)
            score = int(score_str.group()) if score_str else 0
            if score >= GRADE_THRESHOLD:
                relevant_docs.append(docs[i])
            scores.append(score)
        
        relevance_rate = len(relevant_docs) / len(docs)
        if relevance_rate < 0.5:
            reflect_template = PromptTemplate.from_template(
                f"{prompts_text}\nReflect: Query '{{query}}' hat nur {{rate}}% relevante Docs (Scores: {{scores}})."
                "Warum? Vorschlag für besseren Kontext (z.B. Web-Suche)? Antworte kurz."
            )
            reflection = await self.llm.ainvoke(reflect_template.format(query=query, rate=relevance_rate*100, scores=scores))
            if TAVILY_KEY:
                from langchain_community.tools.tavily_search import TavilySearchResults
                fallback = await TavilySearchResults(max_results=3).run(query)
            elif NEWS_KEY:
                from langchain_community.utilities import NewsAPIWrapper
                fallback = await NewsAPIWrapper(news_api_key=NEWS_KEY).run(query)
            else:
                fallback = "Web-Fallback: Kein Key – suche manuell."
            return f"Reflection: {reflection.content}\nFallback: {fallback}"
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = splitter.split_documents(relevant_docs)
        embed_tasks = [self.embeddings.aembed_query(chunk.page_content) for chunk in splits]
        embeds = await asyncio.gather(*embed_tasks)
        
        context = "\n".join([split.page_content for split in splits[:3]])
        gen_template = PromptTemplate.from_template(
            f"{prompts_text}\nQuery: {{query}}\nContext: {{context}}\nAntwort:"
        )
        response = await self.llm.ainvoke(gen_template.format(query=query, context=context))
        
        return f"Adaptive RAG: {len(relevant_docs)}/{len(docs)} Docs. Response: {response.content}"
    
    async def status(self, memory_manager, current_llm_name):
        """Zeige CLI-Status (Modus, LLM, Topic, Memory, RAG)."""
        table = Table(title="CLI Status", show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="cyan")
        table.add_column("Wert", style="white")
        table.add_row("Default-Modus", config["general"]["default_mode"])
        table.add_row("LLM", current_llm_name)
        table.add_row("Topic", memory_manager.current_topic)
        table.add_row("Memory-Typ", memory_manager.current_type)
        table.add_row("Memory-Messages", str(len(memory_manager.memories.get(memory_manager.current_topic, {}).get("instance", {}).chat_memory.messages)))
        vs = Chroma(persist_directory=f"{CHROMA_DIR}/global", embedding_function=self.embeddings)
        table.add_row("RAG-Docs", str(len(vs.get()['ids'])))
        table.add_row("Zeit", datetime.now().strftime('%Y-%m-%d %H:%M'))
        rprint(Panel(table, title="Status", border_style="green"))