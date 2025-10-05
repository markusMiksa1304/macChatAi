import pytest
from pathlib import Path
from core_components import UltimateRAGSetup
from langchain_community.embeddings import HuggingFaceEmbeddings
import tomllib

@pytest.mark.asyncio
async def test_rag_index():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    rag = UltimateRAGSetup(embeddings, None)
    await rag.index_file_from_string("Test content", "test")
    assert Path("chroma_db/test").exists()

@pytest.mark.asyncio
async def test_scrape_playwright():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    rag = UltimateRAGSetup(embeddings, None)
    preview = await rag.scrape_with_playwright("https://quotes.toscrape.com", "test_playwright")
    assert Path("chroma_db/test_playwright").exists()
    assert preview is not None

@pytest.mark.asyncio
async def test_scrape_requests_html():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    rag = UltimateRAGSetup(embeddings, None)
    preview = await rag.scrape_with_requests_html("https://quotes.toscrape.com", "test_requests_html")
    assert Path("chroma_db/test_requests_html").exists()
    assert preview is not None

@pytest.mark.asyncio
async def test_mcp_api():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    rag = UltimateRAGSetup(embeddings, None)
    preview = await rag.call_mcp_api("test", "test_mcp")
    assert Path("chroma_db/test_mcp").exists()
    assert preview is not None

@pytest.mark.asyncio
async def test_auto_fetch():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    rag = UltimateRAGSetup(embeddings, None)
    result, tool_name, confidence = await rag.fetch_and_index("https://quotes.toscrape.com", "test_fetch")
    assert Path("chroma_db/test_fetch").exists()
    assert result is not None
    assert tool_name in ["scrape_url", "scrape_playwright", "scrape_requests_html", "mcp_api", "rss_feed"]
    assert 0 <= confidence <= 1

@pytest.mark.asyncio
async def test_set_mode():
    with open("configs.toml", "r") as f:
        cfg = tomllib.loads(f.read())
    original_mode = cfg["general"]["default_mode"]
    cfg["general"]["default_mode"] = "chat"
    with open("configs.toml", "w") as f:
        tomllib.dump(cfg, f)
    with open("configs.toml", "r") as f:
        cfg = tomllib.loads(f.read())
    assert cfg["general"]["default_mode"] == "chat"
    cfg["general"]["default_mode"] = original_mode
    with open("configs.toml", "w") as f:
        tomllib.dump(cfg, f)