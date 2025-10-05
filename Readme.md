ultimate_mlx_cli_vfinal/
├── configs.toml              # Konfiguration (z. B. default_mode, MCP-API)
├── cli_main.py              # Haupt-CLI mit Selektor und Default-Modus
├── core_components.py       # Memory, RAG, Scraping-Methoden
├── utils.py                 # Tools, Auto-Selektor, Status
├── upload2rag.py            # Indexer (unverändert)
├── rag_worker.py            # Worker (unverändert)
├── requirements.txt         # Abhängigkeiten (unverändert)
├── prompts/                 # Verzeichnis für Prompts
│   └── example_prompt.md    # Beispiel-Prompt (unverändert)
├── tests/                   # Test-Verzeichnis
│   └── test_ultimate_cli_real.py  # Tests für RAG, Fetch, Selektor
├── README.md                # Anleitung und Übersicht
├── chroma_db/               # Verzeichnis für Chroma-Vektoren (wird bei Laufzeit erstellt)
├── upload2rag_docs/         # Verzeichnis für hochgeladene Docs (wird erstellt)
├── logs/                    # Verzeichnis für Logs (wird erstellt)
│   └── dialogs/             # Unterverzeichnis für Dialog-Logs
├── outputs/                 # Verzeichnis für Batch-Ausgaben (wird erstellt)
└── mlx_models/              # Verzeichnis für MLX-Modelle (wird erstellt)

# Ultimate MLX-CLI vFinal

mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ./mlx_models/qwen2.5-7b --quantize --q-bits 4 --q-mode mxfp4

pip install -r requirements.txt
playwright install
mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ./mlx_models/qwen2.5-7b --quantize 4
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management

[apis]
mcp = { endpoint = "https://newsapi.org/v2/everything?q={query}&apiKey={api_key}", api_key = "abc123", method = "GET", headers = { "Authorization" = "Bearer {api_key}" } }

python upload2rag.py &
for i in {1..4}; do python rag_worker.py & done
python cli_main.py

/status
/set_mode fetch
Drohnen München
https://www.sueddeutsche.de/muenchen
https://quotes.toscrape.com
/set_mode chat
Drohnen München
/status

Ausgaben:

/status: Panel mit "Default-Modus: fetch", "LLM: mlx", "Topic: default", "Memory-Typ: buffer", "Memory-Messages: X", "RAG-Docs: Y".
Drohnen München (fetch): "Auto-Fetch (mcp_api, Confidence: 0.90): [News JSON]... (3 Chunks)".
https://www.sueddeutsche.de/muenchen (fetch): "Auto-Fetch (scrape_playwright, Confidence: 0.90): München News... (8 Chunks)".
https://quotes.toscrape.com (fetch): "Auto-Fetch (scrape_requests_html, Confidence: 0.85): Quotes... (5 Chunks)".
/set_mode chat: "Default-Modus auf chat gesetzt."
Drohnen München (chat): "Adaptive RAG: 3/5 Docs. Response: 3–6 Drohnen (Aug 2025), 32 Flüge betroffen."


Performance:

Qwen2.5-7B (MLX): ~35 Tokens/s, ~8 GB RAM.
Auto-Tool: ~0.5–1s, <100 MB RAM.
Playwright: ~2–5s, ~1–2 GB RAM.
Requests-HTML: ~1–2s, ~500 MB RAM.
MCP-API: ~0.5–2s, <100 MB RAM.



Lokale KI-CLI mit MLX (Qwen2.5), RAG, Memory & Tools für Mac Pro (36 GB).

## Features
- LLM: MLX (Qwen2.5-7B, DeepSeek), Ollama (Qwen2.5, Llama-3.1), OpenAI (GPT-4o-mini).
- RAG: PDF-Chunking, Repo-Analyse, Adaptive RAG (Grade/Reflect).
- Scraping: BS4, Scrapy, Playwright (dynamische News), Requests-HTML (einfache Sites), MCP-API (generische API).
- Auto-Tool: `/fetch <query>` wählt automatisch das beste Tool (BS4, Playwright, Requests-HTML, MCP-API, RSS) mit Confidence-Score.
- Default-Modus: `/set_mode <fetch|chat>` setzt Standard-Verhalten (init: fetch), Eingaben ohne `/` werden entsprechend verarbeitet.
- Status: `/status` zeigt Modus, LLM, Topic, Memory, RAG-Stats.
- Memory: Hybride (Buffer/Summary/Entity/KG), Compress/Isolate.
- Tools: YouTube-Transkript, DuckDuckGo, Tavily/NewsAPI, RSS, BS4/Scrapy/Playwright/Requests-HTML/MCP-API.
- Indexer: Watchdog + Parallel (RabbitMQ-Workers).
- Batch: Fragen aus TXT, MD-Export in ./outputs/.
- UI: Rich-Farben, Menü, Kontext (05.10.2025, München, Markus).

## Setup
1. `pip install -r requirements.txt`
2. Install Playwright: `playwright install`
3. Config: `configs.toml` (Keys, Pfade, default_mode = "fetch").
4. Modelle:
   - MLX: `mlx_lm.convert --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ./mlx_models/qwen2.5-7b --quantize 4`
   - Ollama: `ollama pull qwen2.5:7b-instruct`
5. RabbitMQ: `docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management`
6. Starte: `python upload2rag.py` (Indexer), `python rag_worker.py` (Worker), `python cli_main.py` (CLI).

## Usage
- Default: `Drohnen München` → `/fetch Drohnen München` (auto-Tool).
- Set Modus: `/set_mode chat` (Eingaben → `/chat`).
- Status: `/status` (Modus, LLM, Topic, Memory, RAG).
- Chat: `/chat Hallo` oder direkt tippen (bei Modus `chat`).
- Batch: `/batch_questions questions.txt` → MDs in `./outputs/`.
- RSS: `/feed https://rss.nytimes.com/services/xml/rss/nyt/World.xml`.
- Scrape:
  - BS4: `/scrape_url https://www.bbc.com/news`
  - Playwright: `/scrape_playwright https://www.sueddeutsche.de/muenchen`
  - Requests-HTML: `/scrape_requests_html https://quotes.toscrape.com`
  - Auto: `/fetch https://www.sueddeutsche.de/muenchen` (wählt Playwright)
- API: `/call_api Drohnen München` oder `/fetch Drohnen München` (wählt MCP-API).
- Crawl: `/crawl_site https://quotes.toscrape.com 10` (Scrapy).
- YouTube: `/chat Transkript von https://youtu.be/QkdtQ70RFIQ analysieren?`.

## Struktur
- `cli_main.py`: Haupt-CLI (Selektor, Status)
- `core_components.py`: Memory/RAG (BS4/Scrapy/Playwright/Requests-HTML/MCP-API/Fetch)
- `utils.py`: Helpers (Tools, Auto-Selektor)
- `upload2rag.py`: Indexer
- `rag_worker.py`: Worker
- `configs.toml`, `requirements.txt`, `prompts/`, `tests/`

## Troubleshooting
- RabbitMQ: UI localhost:15672 (guest/guest).
- MLX: macOS Sequoia+ für VT-Decoder.
- Playwright: `playwright install`.
- Keys: Tavily (tavily.com), NewsAPI (newsapi.org).
- Video: BS4/Scrapy optimiert nach https://youtu.be/QkdtQ70RFIQ.