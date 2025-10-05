import asyncio
import sys
import subprocess
from pathlib import Path
import tomllib
from rich.console import Console
from rich import print as rprint
from datetime import datetime
import uuid

from core_components import UltimateMemoryManager, UltimateRAGSetup, MLXOptimizer
from utils import ColorfulStatus, build_menu, setup_tools, mcp_api_tool

with open("configs.toml", "rb") as f:
    config = tomllib.load(f)

MODEL_PATH = config["general"]["model_path"]
EMBEDDINGS_MODEL = config["general"]["embeddings_model"]
CHROMA_DIR = config["general"]["chroma_dir"]
UPLOAD_DIR = Path(config["general"]["upload_docs_dir"])
PROMPTS_DIR = Path(config["general"]["prompts_dir"])
LOGS_DIR = Path(config["general"]["logs_dir"])
MEMORY_WINDOW = config["general"]["memory_window"]
REDIS_URL = config["memory"]["redis_url"]
OPENAI_API_KEY = config["apis"]["openai_key"]
TAVILY_KEY = config["apis"]["tavily_key"]
NEWS_KEY = config["apis"]["newsapi_key"]
MLX_QUANT = config["mlx_tuning"]["quant"]
MLX_ATTN = config["mlx_tuning"]["attn_mode"]
MLX_LORA = config["mlx_tuning"]["lora_path"]
MLX_BATCH = config["mlx_tuning"]["batch_size"]
CHUNK_SIZE = config["rag"]["chunk_size"]
CHUNK_OVERLAP = config["rag"]["chunk_overlap"]
PDF_MODE = config["rag"]["pdf_mode"]
REPO_ANALYZE = config["rag"]["repo_analyze"]
ADAPTIVE_RAG = config["rag"]["adaptive_rag"]
GRADE_THRESHOLD = config["rag"]["grade_threshold"]
CONTEXT_ISOLATE = config["memory"]["context_isolate"]
SHOW_MENU_AFTER_CMD = config["general"]["show_menu_after_cmd"]
DEFAULT_MODE = config["general"]["default_mode"]

PROMPTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
Path("./outputs").mkdir(exist_ok=True)

console = Console()

async def main():
    build_menu()
    
    from langchain_community.llms.mlx_pipeline import MLXPipeline
    from langchain_community.chat_models.mlx import ChatMLX
    from langchain_openai import ChatOpenAI
    from langchain_community.llms import Ollama
    
    with ColorfulStatus.loading("MLX Load", "Optimiere Modell..."):
        llm_pipeline_kwargs = {
            "max_tokens": config["mlx_tuning"]["max_tokens"],
            "temp": config["mlx_tuning"]["temp"],
            "top_p": config["mlx_tuning"]["top_p"],
            "top_k": config["mlx_tuning"]["top_k"],
            "batch_size": MLX_BATCH,
            "attn_mode": MLX_ATTN,
            "vt_decoder": config["mlx_tuning"]["vt_decoder"]
        }
        if MLX_LORA:
            llm_pipeline_kwargs["lora_path"] = MLX_LORA
        llm_pipeline = MLXPipeline.from_model_id(MODEL_PATH, pipeline_kwargs=llm_pipeline_kwargs)
        mlx_llm = ChatMLX(llm=llm_pipeline)
    
    openai_llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini") if OPENAI_API_KEY else None
    ollama_llm = Ollama(model=config["general"]["ollama_model"]) if config["general"].get("ollama_model") else None
    current_llm = mlx_llm
    current_llm_name = "mlx"
    
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    
    multi_memory = UltimateMemoryManager(llm=current_llm)
    rag = UltimateRAGSetup(embeddings, current_llm)
    logger = JSONDialogLogger(LOGS_DIR)
    
    from langchain import hub
    from langchain.agents import create_react_agent, AgentExecutor
    tools = setup_tools(current_llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(current_llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    dialog_id = str(uuid.uuid4())
    ColorfulStatus.info(f"Dialog-ID: {dialog_id}")
    current_mode = DEFAULT_MODE
    ColorfulStatus.info(f"Default-Modus: {current_mode}")
    
    while True:
        user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        if user_input.startswith("/"):
            cmd_parts = user_input[1:].split(maxsplit=1)
            cmd_name = cmd_parts[0]
            args = cmd_parts[1] if len(cmd_parts) > 1 else ""
            
            try:
                if cmd_name == "help":
                    build_menu()
                elif cmd_name == "set_mode" and args in ["fetch", "chat"]:
                    current_mode = args
                    with open("configs.toml", "r") as f:
                        cfg = tomllib.loads(f.read())
                    cfg["general"]["default_mode"] = args
                    with open("configs.toml", "w") as f:
                        tomllib.dump(cfg, f)
                    ColorfulStatus.success(f"Default-Modus auf {args} gesetzt.")
                elif cmd_name == "status":
                    await rag.status(multi_memory, current_llm_name)
                elif cmd_name == "switch_llm" and args in ["mlx", "openai", "ollama"]:
                    if args == "mlx":
                        current_llm = mlx_llm
                    elif args == "openai" and openai_llm:
                        current_llm = openai_llm
                    elif args == "ollama" and ollama_llm:
                        current_llm = ollama_llm
                    current_llm_name = args
                    multi_memory.set_llm(current_llm)
                    rag.llm = current_llm
                    ColorfulStatus.success(f"LLM gewechselt zu {args}.")
                    tools = setup_tools(current_llm)
                    agent = create_react_agent(current_llm, tools, prompt)
                    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                elif cmd_name == "opt_mlx" and args:
                    MLXOptimizer.convert_to_mlx_optimized(args, *MLXOptimizer.extract_hf_repo(args))
                elif cmd_name == "extract" and args:
                    repo, template = MLXOptimizer.extract_hf_repo(args)
                    if repo:
                        MLXOptimizer.check_hf_version(repo)
                        MLXOptimizer.convert_to_mlx_optimized(args, repo, template)
                elif cmd_name == "benchmark":
                    MLXOptimizer.benchmark_mlx(MODEL_PATH)
                elif cmd_name == "memory_type" and args in ["buffer", "summary_buffer", "entity", "kg"]:
                    await multi_memory.switch_topic(multi_memory.current_topic, args)
                elif cmd_name == "memory_status":
                    await multi_memory.memory_status()
                elif cmd_name == "clear_memory":
                    await multi_memory.clear_memory(args or None)
                elif cmd_name == "export_memory":
                    await multi_memory.export_memory(args or None)
                elif cmd_name == "load_pdf" and args:
                    path, pages = (args.rsplit(maxsplit=1) if " " in args else (args, None))
                    await rag.add_pdf(path, multi_memory.current_topic, pages)
                elif cmd_name == "arxiv" and args:
                    await rag.search_arxiv(args, multi_memory.current_topic)
                elif cmd_name == "index_docs" and args:
                    await rag.index_file(Path(args))
                    ColorfulStatus.success(f"Indiziert: {args}")
                elif cmd_name == "parallel_index" and args:
                    await rag.parallel_index(args)
                elif cmd_name == "watch_docs" and args == "start":
                    await rag.start_watchdog()
                elif cmd_name == "watch_docs" and args == "stop":
                    await rag.stop_watchdog()
                elif cmd_name == "adaptive_rag" and args:
                    response = await rag.adaptive_rag(args, multi_memory.current_topic)
                    rprint(f"[bold blue]Adaptive RAG:[/bold blue] {response}")
                elif cmd_name == "rag_status":
                    from langchain_community.vectorstores import Chroma
                    vs = Chroma(persist_directory=f"{CHROMA_DIR}/global", embedding_function=embeddings)
                    rprint(f"[green]📊 Indizierte Docs: {len(vs.get()['ids'])}[/green]")
                elif cmd_name == "switch" and args:
                    await multi_memory.switch_topic(args)
                elif cmd_name == "save":
                    await multi_memory.save_topics()
                elif cmd_name == "log_status":
                    rprint(f"[blue]📋 Logs in {LOGS_DIR}. Dateien: {list(LOGS_DIR.glob('*.json'))}[/blue]")
                elif cmd_name == "clear_logs":
                    for f in LOGS_DIR.glob("*.json"):
                        f.unlink()
                    ColorfulStatus.success("Logs gelöscht.")
                elif cmd_name == "test_mode" and args in ["unit", "real"]:
                    subprocess.run(["pytest", "tests/", "-v"])
                    ColorfulStatus.success("Tests gelaufen!")
                elif cmd_name == "batch_questions" and args:
                    questions_file = Path(args)
                    if not questions_file.exists():
                        ColorfulStatus.error("Fragen-TXT nicht gefunden.")
                        continue
                    with open(questions_file, "r") as f:
                        questions = [line.strip() for line in f if line.strip()]
                    async def process_question(q):
                        memory = await multi_memory.get_memory(multi_memory.current_topic)
                        response = await rag.adaptive_rag(q, multi_memory.current_topic) if ADAPTIVE_RAG else "Fallback-Response"
                        return f"# Frage: {q}\n\n## Antwort: {response}\n\n---\n"
                    batch_size = 4
                    all_responses = []
                    for i in range(0, len(questions), batch_size):
                        batch = questions[i:i+batch_size]
                        batch_responses = await asyncio.gather(*[process_question(q) for q in batch])
                        all_responses.extend(batch_responses)
                    outputs_dir = Path("./outputs")
                    for idx, md_content in enumerate(all_responses):
                        with open(outputs_dir / f"frage_{idx+1}-antwort.md", "w") as f:
                            f.write(md_content)
                    ColorfulStatus.success(f"Batch abgeschlossen: {len(questions)} MDs in ./outputs/")
                elif cmd_name == "feed" and args:
                    from utils import rss_feed_tool
                    feed_data = rss_feed_tool(args)
                    await rag.index_file_from_string(feed_data, "feed_topic")
                    rprint(f"[blue]Feed parsed: {feed_data}[/blue]")
                elif cmd_name == "scrape_url" and args:
                    preview = await rag.scrape_and_index_url(args, multi_memory.current_topic)
                    rprint(f"[green]BS4 Scraped: {preview}[/green]")
                elif cmd_name == "scrape_playwright" and args:
                    preview = await rag.scrape_with_playwright(args, multi_memory.current_topic)
                    rprint(f"[green]Playwright Scraped: {preview}[/green]")
                elif cmd_name == "scrape_requests_html" and args:
                    preview = await rag.scrape_with_requests_html(args, multi_memory.current_topic)
                    rprint(f"[green]Requests-HTML Scraped: {preview}[/green]")
                elif cmd_name == "call_api" and args:
                    preview = await rag.call_mcp_api(args, multi_memory.current_topic)
                    rprint(f"[green]MCP-API Response: {preview}[/green]")
                elif cmd_name == "fetch" and args:
                    result, tool_name, confidence = await rag.fetch_and_index(args, multi_memory.current_topic)
                    rprint(f"[green]Auto-Fetch ({tool_name}, Confidence: {confidence:.2f}): {result}[/green]")
                elif cmd_name == "crawl_site" and args:
                    start_url, max_pages = (args.split(maxsplit=1) if " " in args else (args, "10"))
                    max_pages = int(max_pages)
                    await rag.crawl_and_index_site(start_url, max_pages, multi_memory.current_topic)
                    ColorfulStatus.success(f"Crawl abgeschlossen: {start_url}")
                elif cmd_name == "start_workers" and args:
                    count = int(args)
                    for _ in range(count):
                        subprocess.Popen(["python", "rag_worker.py"])
                    ColorfulStatus.success(f"{count} Workers gestartet.")
                elif cmd_name == "exit":
                    await multi_memory.save_topics()
                    await rag.stop_watchdog()
                    sys.exit(0)
                else:
                    ColorfulStatus.warn(f"Unbekannter Befehl: {cmd_name}. /help für Liste.")
                    build_menu()
                
                if SHOW_MENU_AFTER_CMD:
                    build_menu()
            except Exception as e:
                ColorfulStatus.error(f"Fehler: {e}")
                build_menu()
            continue
        
        if not user_input:
            continue
        
        memory = await multi_memory.get_memory(multi_memory.current_topic)
        prompts_text = rag.load_external_prompts(PROMPTS_DIR)
        api_context = ""
        if MCP_CONFIG and user_input.lower().startswith(("wie viele drohnen", "news", "aktuell")):
            api_context = mcp_api_tool(user_input)[:500]
        system_context = (
            f"Heute ist {datetime.now().strftime('%Y-%m-%d %H:%M')} und Ort (München) "
            f"du bist auf macPro 36gb , Partner Markus\n{prompts_text}\n"
            f"API-Daten: {api_context}\nDefault-Modus: {current_mode}"
        )
        prefill_prompt = f"{system_context}\nContext: {memory.load_memory_variables({})['chat_history']}\nUser: {user_input}\nAI:"
        
        if current_mode == "fetch":
            result, tool_name, confidence = await rag.fetch_and_index(user_input, multi_memory.current_topic)
            response = f"Auto-Fetch ({tool_name}, Confidence: {confidence:.2f}): {result}"
            rprint(f"[bold yellow]AI ({current_llm_name}):[/bold yellow] {response}\n")
        else:  # chat
            if ADAPTIVE_RAG:
                response = await rag.adaptive_rag(user_input, multi_memory.current_topic)
            else:
                from langchain.chains import ConversationalRetrievalChain
                retriever = rag.get_retriever(multi_memory.current_topic)
                if retriever:
                    conv_chain = ConversationalRetrievalChain.from_llm(current_llm, retriever, memory=memory)
                    response = conv_chain.invoke({"question": user_input})["answer"]
                else:
                    response = agent_executor.invoke({"input": user_input})["output"]
            rprint(f"[bold yellow]AI ({current_llm_name}):[/bold yellow] {response}\n")
        
        memory.save_context({"input": user_input}, {"output": response})
        llm_call = {"model": current_llm_name, "response": response, "tokens": len(response.split())}
        memory_state = {"messages": len(memory.chat_memory.messages)}
        logger.log_turn(dialog_id, user_input, prefill_prompt, llm_call, [], memory_state)