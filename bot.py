#!/usr/bin/env python3
"""Telegram + Ollama + MCP + Open Interpreter + ChromaDB 비동기 멀티 에이전트 봇."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import os
import time
import traceback
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import chromadb
import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv


# ---------------------------------------------------------
# 로깅
# ---------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mega_ai_bot")


# ---------------------------------------------------------
# 모델 프로필
# ---------------------------------------------------------
@dataclass(frozen=True)
class ModelProfile:
    name: str
    keep_alive: Any
    role: str


MODEL_PROFILES: Dict[str, ModelProfile] = {
    "fast": ModelProfile(name="llama3.1:8b", keep_alive=-1, role="빠른 일반 대화"),
    "general": ModelProfile(name="llama3.1:8b", keep_alive=-1, role="일반 지식/작성"),
    "verifier": ModelProfile(name="qwen3.5:35b", keep_alive=0, role="검토/비평"),
    "code": ModelProfile(name="qwen3-coder:30b", keep_alive=0, role="코드/디버깅"),
    "reason": ModelProfile(name="deepseek-r1:32b", keep_alive=0, role="고난도 추론"),
    "synth": ModelProfile(name="gemma2:9b", keep_alive=-1, role="최종 합성"),
    "fallback": ModelProfile(name="phi3:latest", keep_alive=0, role="경량 대체"),
}


@dataclass
class Settings:
    telegram_bot_token: str
    ollama_base_url: str = "http://127.0.0.1:11434"
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "conversation_memory"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    pipeline_timeout_sec: float = 1800.0

    enable_open_interpreter: bool = False
    open_interpreter_model: str = "ollama/llama3.1:8b"
    open_interpreter_api_base: str = "http://127.0.0.1:11434"

    mcp_servers_json: str = "[]"
    workspace_dir: str = "./workspace_steps"
    max_stage_chars: int = 1400


def load_settings() -> Settings:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN 환경변수가 필요합니다.")

    return Settings(
        telegram_bot_token=token,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        chroma_path=os.getenv("CHROMA_PATH", "./chroma_db"),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "conversation_memory"),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        ),
        pipeline_timeout_sec=float(os.getenv("PIPELINE_TIMEOUT_SEC", "1800")),
        enable_open_interpreter=os.getenv("ENABLE_OPEN_INTERPRETER", "false").lower() in {"1", "true", "yes", "on"},
        open_interpreter_model=os.getenv("OI_MODEL", "ollama/llama3.1:8b"),
        open_interpreter_api_base=os.getenv("OI_API_BASE", "http://127.0.0.1:11434"),
        mcp_servers_json=os.getenv("MCP_SERVERS_JSON", "[]"),
        workspace_dir=os.getenv("CHAIN_WORKSPACE_DIR", "./workspace_steps"),
        max_stage_chars=int(os.getenv("MULTI_MAX_CHARS_PER_STAGE", "1400")),
    )


# ---------------------------------------------------------
# 유틸
# ---------------------------------------------------------
def summarize_text(text: str, max_chars: int) -> str:
    t = text.strip()
    return t if len(t) <= max_chars else f"{t[:max_chars]}\n...(중간 출력 생략)"


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)


# ---------------------------------------------------------
# Ollama Client (timeout=None)
# ---------------------------------------------------------
class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=None)

    async def close(self) -> None:
        await self.client.aclose()

    @staticmethod
    def normalize_keep_alive(value: Any) -> Any:
        if isinstance(value, str) and value.strip() in {"-1", "0"}:
            return int(value.strip())
        return value

    async def chat(
        self,
        model: ModelProfile,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model.name,
            "stream": False,
            "messages": messages,
            "options": {"temperature": temperature},
            "keep_alive": self.normalize_keep_alive(model.keep_alive),
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = await self.client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.exception("Ollama HTTP 오류")
            raise RuntimeError(f"Ollama bad response: {e.response.text}") from e
        except Exception as e:
            logger.exception("Ollama 호출 실패")
            raise RuntimeError(f"Ollama error: {e}") from e


# ---------------------------------------------------------
# ChromaDB Memory
# ---------------------------------------------------------
class MemoryStore:
    def __init__(self, settings: Settings):
        emb_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model_name)
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"},
        )

    async def add_turn(self, chat_id: int, role: str, text: str) -> None:
        doc = f"[{role}] {text.strip()}"
        ts = datetime.now(timezone.utc).isoformat()
        await asyncio.to_thread(
            self.collection.add,
            ids=[str(uuid.uuid4())],
            documents=[doc],
            metadatas=[{"chat_id": str(chat_id), "role": role, "ts": ts}],
        )

    async def retrieve(self, chat_id: int, query: str, k: int = 5) -> List[str]:
        try:
            res = await asyncio.to_thread(
                self.collection.query,
                query_texts=[query],
                n_results=k,
                where={"chat_id": str(chat_id)},
            )
            return res.get("documents", [[]])[0] or []
        except Exception:
            logger.warning("Chroma 검색 실패", exc_info=True)
            return []


# ---------------------------------------------------------
# Tool 인터페이스
# ---------------------------------------------------------
class ToolBase:
    name: str
    description: str
    input_schema: Dict[str, Any]

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    async def run(self, arguments: Dict[str, Any]) -> str:
        raise NotImplementedError


class OpenInterpreterTool(ToolBase):
    name = "open_interpreter"
    description = "Open Interpreter로 실제 시스템 작업을 수행합니다."
    input_schema = {
        "type": "object",
        "properties": {"instruction": {"type": "string"}},
        "required": ["instruction"],
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self._interpreter: Any = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            module = importlib.import_module("interpreter")
            self._interpreter = module.interpreter
            self._interpreter.auto_run = True
            self._interpreter.llm.model = "ollama/llama3.1:8b"
            self._interpreter.llm.api_base = "http://127.0.0.1:11434"
            if getattr(self._interpreter.llm, "api_key", None) in {None, ""}:
                self._interpreter.llm.api_key = "x"
        except Exception:
            logger.warning("Open Interpreter 초기화 실패", exc_info=True)
            self._interpreter = None

    async def run(self, arguments: Dict[str, Any]) -> str:
        if self._interpreter is None:
            return "Open Interpreter 사용 불가(초기화 실패)."
        instruction = str(arguments.get("instruction", "")).strip()
        if not instruction:
            return "instruction 인자가 비어 있습니다."

        def _sync() -> str:
            result = self._interpreter.chat(instruction)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)

        try:
            return await asyncio.to_thread(_sync)
        except Exception as e:
            return f"Open Interpreter 실행 실패: {e}"


class MCPTool(ToolBase):
    def __init__(self, server_name: str, tool_name: str, description: str, input_schema: Dict[str, Any], call_fn: Callable):
        self.name = f"mcp::{server_name}::{tool_name}"
        self.description = description or f"MCP tool {tool_name}"
        self.input_schema = input_schema or {"type": "object", "properties": {}}
        self._call_fn = call_fn

    async def run(self, arguments: Dict[str, Any]) -> str:
        try:
            result = await self._call_fn(arguments)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"MCP tool 실행 실패: {e}"


class MCPManager:
    def __init__(self, server_configs: List[Dict[str, Any]]):
        self.server_configs = server_configs
        self.tools: List[MCPTool] = []
        self.exit_stack: Optional[AsyncExitStack] = None

    async def connect_and_discover(self) -> List[MCPTool]:
        self.tools = []
        if not self.server_configs:
            logger.info("MCP 서버 설정이 없어 건너뜁니다.")
            return []

        try:
            stdio_mod = importlib.import_module("mcp.client.stdio")
            session_mod = importlib.import_module("mcp.client.session")
            mcp_mod = importlib.import_module("mcp")
            stdio_client = getattr(stdio_mod, "stdio_client")
            ClientSession = getattr(session_mod, "ClientSession")
            StdioServerParameters = getattr(mcp_mod, "StdioServerParameters", None)
        except Exception:
            logger.warning("MCP SDK import 실패로 MCP 기능 비활성화")
            return []

        self.exit_stack = AsyncExitStack()

        try:
            for cfg in self.server_configs:
                name = cfg.get("name", "unknown")
                command = cfg.get("command")
                args = cfg.get("args", [])
                env = cfg.get("env")
                if not command:
                    logger.warning("MCP 서버(%s) command 누락", name)
                    continue

                try:
                    stdio_cm = self._build_stdio_context(stdio_client, StdioServerParameters, command, args, env)
                    read_stream, write_stream = await self.exit_stack.enter_async_context(stdio_cm)
                    session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
                    await session.initialize()
                    tool_list = await session.list_tools()

                    for t in tool_list.tools:
                        async def _call_tool(arguments: Dict[str, Any], s=session, tn=t.name):
                            res = await s.call_tool(tn, arguments)
                            return getattr(res, "content", str(res))

                        self.tools.append(
                            MCPTool(
                                server_name=name,
                                tool_name=t.name,
                                description=getattr(t, "description", ""),
                                input_schema=getattr(t, "inputSchema", {"type": "object", "properties": {}}),
                                call_fn=_call_tool,
                            )
                        )
                    logger.info("MCP 서버 연결 성공: %s (tools=%d)", name, len(tool_list.tools))
                except Exception:
                    logger.exception("MCP 서버 연결 실패(건너뜀): %s", name)
        except Exception:
            logger.exception("MCP 초기화 실패, 전체 MCP 비활성화")
            await self.close()

        return self.tools

    @staticmethod
    def _build_stdio_context(stdio_client: Callable, StdioServerParameters: Any, command: str, args: List[str], env: Optional[Dict[str, str]]):
        sig = inspect.signature(stdio_client)
        params = set(sig.parameters.keys())

        if StdioServerParameters is not None and "server_parameters" in params:
            return stdio_client(server_parameters=StdioServerParameters(command=command, args=args, env=env))
        if StdioServerParameters is not None and "server" in params:
            return stdio_client(server=StdioServerParameters(command=command, args=args, env=env))
        if "command" in params:
            return stdio_client(command=command, args=args, env=env)
        return stdio_client(command, args, env)

    async def close(self) -> None:
        if self.exit_stack is None:
            return
        try:
            await self.exit_stack.aclose()
        except Exception:
            logger.warning("MCP 종료 오류(무시)", exc_info=True)
        finally:
            self.exit_stack = None


# ---------------------------------------------------------
# Multi-agent orchestrator
# ---------------------------------------------------------
class MultiAgentOrchestrator:
    def __init__(self, settings: Settings, ollama: OllamaClient, memory: MemoryStore, tools: List[ToolBase]):
        self.settings = settings
        self.ollama = ollama
        self.memory = memory
        self.tools = {tool.name: tool for tool in tools}

    async def _planner_with_tools(self, chat_id: int, task: str) -> Tuple[str, str, List[Dict[str, Any]]]:
        history = await self.memory.retrieve(chat_id, task, k=5)
        memory_context = "\n".join(history) if history else "(기억 없음)"
        tool_trace: List[Dict[str, Any]] = []

        msgs: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "당신은 Planner입니다. 도구가 꼭 필요한 경우에만 tool_calls를 사용하세요.\n"
                    "출력은 마지막에 한국어 단계 계획서로 정리하세요."
                ),
            },
            {"role": "user", "content": f"[기억]\n{memory_context}\n\n[요청]\n{task}"},
        ]

        schemas = [t.schema() for t in self.tools.values()]

        for _ in range(6):
            resp = await self.ollama.chat(MODEL_PROFILES["fast"], msgs, tools=schemas, temperature=0.1)
            msg = resp.get("message", {})
            tool_calls = msg.get("tool_calls", []) or []
            if not tool_calls:
                plan = msg.get("content", "계획 생성 실패")
                return plan, memory_context, tool_trace

            msgs.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls})
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name")
                raw_args = fn.get("arguments", {})
                args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
                if name not in self.tools:
                    result = f"도구 없음: {name}"
                else:
                    result = await self.tools[name].run(args)
                tool_trace.append({"tool": name, "args": args, "result": result})
                msgs.append({"role": "tool", "name": name, "content": result})

        return "도구 루프 한도 초과", memory_context, tool_trace

    async def run(self, chat_id: int, task: str, specialist_key: str, progress_cb: Callable[[str], Any]) -> str:
        planner_model = MODEL_PROFILES["fast"]
        verifier_model = MODEL_PROFILES["verifier"]
        synthesizer_model = MODEL_PROFILES["synth"]
        specialist = MODEL_PROFILES.get(specialist_key, MODEL_PROFILES["fast"])

        request_id = f"{int(time.time())}_{chat_id}_{uuid.uuid4().hex[:8]}"
        Path(self.settings.workspace_dir).mkdir(parents=True, exist_ok=True)

        plan, memory_context, tool_trace = await self._planner_with_tools(chat_id, task)
        await progress_cb(f"진행중 1/4: 계획 수립 완료 ({planner_model.name})")
        self._save_stage(request_id, 1, "plan", planner_model.name, plan)

        spec_prompt = (
            f"[사용자 요청]\n{task}\n\n"
            f"[기억]\n{memory_context}\n\n"
            f"[계획]\n{summarize_text(plan, self.settings.max_stage_chars)}\n\n"
            f"[도구 로그]\n{summarize_text(json.dumps(tool_trace, ensure_ascii=False, indent=2), self.settings.max_stage_chars)}"
        )
        spec = await self.ollama.chat(specialist, [{"role": "user", "content": spec_prompt}], temperature=0.25)
        spec_text = spec.get("message", {}).get("content", "초안 실패")
        await progress_cb(f"진행중 2/4: 전문가 초안 완료 ({specialist.name})")
        self._save_stage(request_id, 2, "specialist", specialist.name, spec_text)

        review_prompt = (
            f"[요청]\n{task}\n\n"
            f"[초안]\n{summarize_text(spec_text, self.settings.max_stage_chars)}\n\n"
            "문제점/근거/개선안을 한국어로 간결히 작성하세요."
        )
        review = await self.ollama.chat(verifier_model, [{"role": "user", "content": review_prompt}], temperature=0.1)
        review_text = review.get("message", {}).get("content", "검토 실패")
        await progress_cb(f"진행중 3/4: 검토 완료 ({verifier_model.name})")
        self._save_stage(request_id, 3, "review", verifier_model.name, review_text)

        synth_prompt = (
            f"[요청]\n{task}\n\n"
            f"[계획]\n{summarize_text(plan, self.settings.max_stage_chars)}\n\n"
            f"[초안]\n{summarize_text(spec_text, self.settings.max_stage_chars)}\n\n"
            f"[검토]\n{summarize_text(review_text, self.settings.max_stage_chars)}\n\n"
            "최종 답변 형식: 1) 핵심요약 2) 실행단계 3) 주의사항"
        )
        final = await self.ollama.chat(synthesizer_model, [{"role": "user", "content": synth_prompt}], temperature=0.25)
        final_text = final.get("message", {}).get("content", "최종 합성 실패")
        await progress_cb(f"진행중 4/4: 최종 합성 완료 ({synthesizer_model.name})")
        self._save_stage(request_id, 4, "synthesis", synthesizer_model.name, final_text)

        return final_text

    def _save_stage(self, req_id: str, idx: int, stage: str, model_name: str, content: str) -> None:
        file_path = Path(self.settings.workspace_dir) / f"{req_id}_{idx:02d}_{stage}_{safe_name(model_name)}.md"
        file_path.write_text(content.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------
# Telegram app
# ---------------------------------------------------------
class TelegramAIAssistantApp:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bot = Bot(token=settings.telegram_bot_token)
        self.dp = Dispatcher()

        self.ollama = OllamaClient(settings.ollama_base_url)
        self.memory = MemoryStore(settings)

        self.mcp_manager = MCPManager(self._parse_mcp_servers(settings.mcp_servers_json))
        self.orchestrator: Optional[MultiAgentOrchestrator] = None

        self.dp.message.register(self.on_start, Command("start"))
        self.dp.message.register(self.on_models, Command("models"))
        self.dp.message.register(self.on_status, Command("status"))
        self.dp.message.register(self.on_code, Command("code"))
        self.dp.message.register(self.on_reason, Command("reason"))
        self.dp.message.register(self.on_fast, Command("fast"))
        self.dp.message.register(self.on_general, Command("general"))
        self.dp.message.register(self.on_default, F.text)

    @staticmethod
    def _parse_mcp_servers(raw: str) -> List[Dict[str, Any]]:
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else []
        except Exception:
            logger.warning("MCP_SERVERS_JSON 파싱 실패")
            return []

    async def setup(self) -> None:
        tools: List[ToolBase] = []
        if self.settings.enable_open_interpreter:
            tools.append(OpenInterpreterTool(self.settings))
        mcp_tools = await self.mcp_manager.connect_and_discover()
        tools.extend(mcp_tools)

        self.orchestrator = MultiAgentOrchestrator(self.settings, self.ollama, self.memory, tools)
        logger.info("앱 초기화 완료 (tools=%d)", len(tools))

    async def shutdown(self) -> None:
        await self.mcp_manager.close()
        await self.ollama.close()
        await self.bot.session.close()

    async def on_start(self, message: Message) -> None:
        text = (
            "로컬 멀티 AI 비서가 시작되었습니다.\n"
            "- /fast, /general, /code, /reason\n"
            "- /models, /status\n"
            "일반 텍스트는 fast 모드로 처리합니다."
        )
        await message.answer(text)

    async def on_models(self, message: Message) -> None:
        lines = ["사용 가능한 모델 프로필:"]
        for k, v in MODEL_PROFILES.items():
            lines.append(f"- {k}: {v.name} | {v.role}")
        await message.answer("\n".join(lines))

    async def on_status(self, message: Message) -> None:
        enabled = "ON" if self.settings.enable_open_interpreter else "OFF"
        await message.answer(
            f"상태: 정상\n- Ollama: {self.settings.ollama_base_url}\n- Open Interpreter: {enabled}\n- Pipeline timeout: {self.settings.pipeline_timeout_sec:.0f}s"
        )

    async def on_code(self, message: Message) -> None:
        task = (message.text or "").replace("/code", "", 1).strip()
        await self._dispatch_request(message, task, specialist_key="code")

    async def on_reason(self, message: Message) -> None:
        task = (message.text or "").replace("/reason", "", 1).strip()
        await self._dispatch_request(message, task, specialist_key="reason")

    async def on_fast(self, message: Message) -> None:
        task = (message.text or "").replace("/fast", "", 1).strip()
        await self._dispatch_request(message, task, specialist_key="fast")

    async def on_general(self, message: Message) -> None:
        task = (message.text or "").replace("/general", "", 1).strip()
        await self._dispatch_request(message, task, specialist_key="general")

    async def on_default(self, message: Message) -> None:
        task = (message.text or "").strip()
        await self._dispatch_request(message, task, specialist_key="fast")

    async def _dispatch_request(self, message: Message, task: str, specialist_key: str) -> None:
        if not task:
            await message.answer("질문 내용을 함께 보내주세요. 예: /code FastAPI 미들웨어 예시")
            return
        if self.orchestrator is None:
            await message.answer("초기화 중입니다. 잠시 후 재시도해주세요.")
            return

        status = await message.answer(
            "요청을 처리 중입니다... (멀티 에이전트 파이프라인 실행)\n"
            "진행 상태를 순차 업데이트합니다."
        )
        asyncio.create_task(self._process_task(message, status.message_id, task, specialist_key))

    async def _process_task(self, message: Message, status_msg_id: int, task: str, specialist_key: str) -> None:
        assert self.orchestrator is not None
        chat_id = message.chat.id
        msg_id = message.message_id
        start = time.time()

        async def progress(text: str) -> None:
            try:
                await self.bot.edit_message_text(text=text, chat_id=chat_id, message_id=status_msg_id)
            except Exception:
                logger.warning("진행 상태 메시지 업데이트 실패", exc_info=True)

        try:
            await self.memory.add_turn(chat_id, "user", task)
            final = await asyncio.wait_for(
                self.orchestrator.run(chat_id=chat_id, task=task, specialist_key=specialist_key, progress_cb=progress),
                timeout=self.settings.pipeline_timeout_sec,
            )
            await progress(f"완료: {time.time() - start:.1f}초\n엔진: Ollama 멀티 모델 순차 협업")
            await message.answer(final, reply_to_message_id=msg_id)
            await self.memory.add_turn(chat_id, "assistant", final)
        except asyncio.TimeoutError:
            await progress("실패: 파이프라인 시간 초과")
            await message.answer("처리 시간이 너무 길어 중단되었습니다. 질문을 더 작게 나눠 주세요.", reply_to_message_id=msg_id)
        except Exception as e:
            logger.exception("요청 처리 실패")
            await progress("실패: 멀티 에이전트 파이프라인 오류")
            fallback = MODEL_PROFILES["fallback"]
            try:
                resp = await self.ollama.chat(
                    fallback,
                    [{"role": "user", "content": task}],
                    temperature=0.25,
                )
                text = resp.get("message", {}).get("content", "(fallback 응답 없음)")
                await message.answer(f"(fallback:{fallback.name})\n\n{text}", reply_to_message_id=msg_id)
            except Exception as e2:
                await message.answer(f"요청 처리 실패: {e}\nfallback 실패: {e2}", reply_to_message_id=msg_id)
                logger.debug(traceback.format_exc())

    async def run(self) -> None:
        await self.setup()
        try:
            await self.dp.start_polling(self.bot)
        finally:
            await self.shutdown()


async def main() -> None:
    app = TelegramAIAssistantApp(load_settings())
    await app.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("사용자 중단으로 종료합니다.")
