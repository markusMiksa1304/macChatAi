import json
import re
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn
from langchain_community.tools import (
    DuckDuckGoSearchRun, TavilySearchResults, YouTubeSearchTool, GithubAPIWrapper,
    WikipediaQueryRun, ArxivAPIWrapper, NewsAPIWrapper
)
from langchain_community.utilities import OpenWeatherMapAPIWrapper, RequestsWrapper
import tomllib
import feedparser
from bs4 import BeautifulSoup
import requests
from playwright.sync_api import sync_playwright
from requests_html import HTMLSession

console = Console()

with open("configs.toml", "rb") as f:
    config = tomllib.load(f)

TAVILY_KEY = config["apis"]["tavily_key"]
NEWS_KEY = config["apis"]["newsapi_key"]
MCP_CONFIG = config["apis"].get("mcp", {})

class ColorfulStatus:
    @staticmethod
    def success(msg): rprint(f"[green]✓ {msg}[/green]")
    @staticmethod
    def error(msg): rprint(f"[red]✗ {msg}[/red]")
    @staticmethod
    def info(msg): rprint(f"[blue]ℹ {msg}[/blue]")
    @staticmethod
    def warn(msg): rprint(f"[yellow]⚠ {msg}[/yellow]")
    @staticmethod
    def loading(task, msg):
        with Progress(SpinnerColumn(), TextColumn(msg), console=console) as p:
            yield p.add_task(task, total=None)

def build_menu():
    table = Table(title="Ultimate MLX-CLI vFinal", show_header=True, header_style="bold magenta")
    table.add_column("Befehl", style="cyan")
    table.add_column("Beschreibung", style="white")
    commands = {
        "/help": "Zeige Menü",
        "/chat <text>": "Chat (mit Adaptive RAG)",
        "/set_mode <fetch|chat>": "Setze Default-Modus (fetch, chat)",  # Neu
        "/status": "Zeige Modus, LLM, Topic, Memory, RAG-Stats",  # Neu
        "/switch_llm <mlx|openai|ollama>": "Wechsle LLM (z.B. Qwen2.5)",
        "/opt_mlx <ollama>": "MLX-Opt (LoRA/Flash)",
        "/extract <ollama>": "HF-Repo-Extract",
        "/benchmark": "Tokens/s-Test",
        "/memory_type <type>": "Memory-Typ wechsle",
        "/memory_status": "Memory-Status",
        "/clear_memory [topic]": "Memory löschen",
        "/export_memory [topic]": "Export JSON",
        "/load_pdf <path> [pages]": "PDF laden (hi_res Chunk)",
        "/arxiv <query>": "ArXiv-Suche",
        "/index_docs <path>": "Manuell indexen",
        "/parallel_index <dir>": "Batch-Index (Parallel)",
        "/watch_docs <start|stop>": "Watchdog",
        "/adaptive_rag <query>": "Adaptive RAG (Route/Reflect)",
        "/rag_status": "RAG-Stats",
        "/switch <topic>": "Topic wechsle",
        "/save": "Speichern",
        "/log_status": "Logs zeigen",
        "/clear_logs": "Logs löschen",
        "/batch_questions <txt>": "Batch-Fragen, MD-Export",
        "/feed <rss_url>": "RSS-Feed parsen",
        "/scrape_url <url>": "Webseite scrapen (BS4)",
        "/scrape_playwright <url>": "Scrape dynamische Seite (Playwright)",
        "/scrape_requests_html <url>": "Scrape mit JS (Requests-HTML)",
        "/call_api <query>": "Generische API aufrufen (MCP)",
        "/fetch <query>": "Automatische Tool-Auswahl (Web/API)",
        "/crawl_site <url> <max_pages>": "Site crawlen (Scrapy)",
        "/start_workers <count>": "Starte Workers",
        "/test_mode <unit|real>": "Pytest-Run",
        "/exit": "Beenden"
    }
    for cmd, desc in commands.items():
        table.add_row(cmd, desc)
    console.print(Panel(table, title="Commands", border_style="green"))

def setup_tools(llm):
    from langchain_core.tools import Tool
    tools = [DuckDuckGoSearchRun().as_tool()]
    if TAVILY_KEY:
        tools.append(TavilySearchResults(max_results=5).as_tool())
    else:
        ColorfulStatus.warn("Tavily skipped – set tavily_key in configs.toml")
    if NEWS_KEY:
        tools.append(NewsAPIWrapper(news_api_key=NEWS_KEY).as_tool())
    else:
        ColorfulStatus.warn("NewsAPI skipped – set newsapi_key in configs.toml")
    if MCP_CONFIG:
        tools.append(Tool(
            name="mcp_api",
            description="Rufe eine generische API auf für Echtzeit-Daten (z.B. News, Drohnen) aus configs.toml",
            func=lambda x: mcp_api_tool(x)
        ))
    else:
        ColorfulStatus.warn("MCP-API skipped – set mcp in configs.toml")
    tools.extend([
        YouTubeSearchTool().as_tool(),
        GithubAPIWrapper().as_tool(),
        WikipediaQueryRun().as_tool(),
        ArxivAPIWrapper().as_tool(),
        OpenWeatherMapAPIWrapper().as_tool(),
        Tool(name="rss_feed", description="Parse RSS-Feed", func=rss_feed_tool),
        Tool(name="scrape_url", description="Scrape statische Webseite mit BeautifulSoup", func=scrape_url_tool),
        Tool(name="scrape_playwright", description="Scrape dynamische Webseite mit JavaScript (z.B. News)", func=scrape_playwright_tool),
        Tool(name="scrape_requests_html", description="Scrape halb-dynamische Webseite mit JS (z.B. Quotes)", func=scrape_requests_html_tool)
    ])
    return tools

def rss_feed_tool(url):
    feed = feedparser.parse(url)
    entries = []
    for entry in feed.entries[:10]:
        entries.append({
            "title": entry.title,
            "link": entry.link,
            "summary": entry.summary[:200] + "..." if entry.summary else ""
        })
    return json.dumps(entries, indent=2)

def scrape_url_tool(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})
        soup = BeautifulSoup(response.content, 'lxml')
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        content = "\n".join([elem.get_text(strip=True) for elem in soup.select('.content, .main, .article, .product')])
        return content[:200] + "..." if content else soup.get_text(separator='\n', strip=True)[:200] + "..."
    except:
        return "Scrap-Fehler"

def scrape_playwright_tool(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(1000)
            content = page.content()
            browser.close()
        soup = BeautifulSoup(content, 'lxml')
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        content = "\n".join([elem.get_text(strip=True) for elem in soup.select('.content, .main, .article, .product')])
        return content[:200] + "..." if content else soup.get_text(separator='\n', strip=True)[:200] + "..."
    except Exception as e:
        return f"Playwright-Fehler: {e}"

def scrape_requests_html_tool(url):
    try:
        session = HTMLSession()
        response = session.get(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
        response.html.render(timeout=30, sleep=1)
        soup = BeautifulSoup(response.html.html, 'lxml')
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        content = "\n".join([elem.get_text(strip=True) for elem in soup.select('.content, .main, .article, .product')])
        session.close()
        return content[:200] + "..." if content else soup.get_text(separator='\n', strip=True)[:200] + "..."
    except Exception as e:
        return f"Requests-HTML-Fehler: {e}"

def mcp_api_tool(query):
    try:
        if not MCP_CONFIG:
            return "MCP-API-Konfig fehlt in configs.toml"
        wrapper = RequestsWrapper(headers={k: v.format(api_key=MCP_CONFIG.get("api_key", "")) for k, v in MCP_CONFIG.get("headers", {}).items()})
        endpoint = MCP_CONFIG["endpoint"].format(query=query)
        method = MCP_CONFIG.get("method", "GET").upper()
        if method == "GET":
            response = wrapper.get(endpoint)
        elif method == "POST":
            response = wrapper.post(endpoint, data={"query": query})
        else:
            return "Ungültige Methode: Nur GET/POST unterstützt"
        return json.dumps(response.json(), indent=2)[:500] + "..."
    except Exception as e:
        return f"MCP-API-Fehler: {e}"

def auto_tool_selector(query: str, llm=None):
    """Automatische Tool-Auswahl basierend auf Query."""
    if re.match(r'https?://', query):
        try:
            response = requests.head(query, timeout=5)
            content_type = response.headers.get('content-type', '')
            if 'javascript' in content_type.lower() or 'news' in query.lower():
                return 'scrape_playwright', 0.9
            elif 'quote' in query.lower() or 'blog' in query.lower():
                return 'scrape_requests_html', 0.85
            else:
                return 'scrape_url', 0.8
        except:
            return 'scrape_url', 0.7
    elif any(keyword in query.lower() for keyword in ['news', 'drohnen', 'aktuell', 'api']):
        return 'mcp_api', 0.9
    elif 'rss' in query.lower():
        return 'rss_feed', 0.95
    else:
        if llm:
            prompt = (
                f"Query: {query}\n"
                "Wähle das beste Tool aus: scrape_url (statische Webseiten), "
                "scrape_playwright (dynamische News), scrape_requests_html (halb-dynamische Sites), "
                "mcp_api (Echtzeit-Daten), rss_feed (RSS-Feeds).\n"
                "Antworte mit JSON: {\"tool\": \"name\", \"confidence\": 0.0}"
            )
            response = llm.invoke(prompt)
            try:
                result = json.loads(response.content)
                return result['tool'], result.get('confidence', 0.7)
            except:
                return 'scrape_url', 0.7
        return 'scrape_url', 0.7