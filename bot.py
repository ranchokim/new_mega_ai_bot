import asyncio
import json
import logging
import os
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# MCP SDK(공식)와 Open Interpreter는 설치/환경에 따라 import 경로가 달라질 수 있어
# 최대한 안전하게 import하고, 실패 시 기능을 비활성화(fallback)합니다.
try:
    from mcp.client.stdio import stdio_client  # type: ignore
    from mcp.client.session import ClientSession  # type: ignore
except Exception:  # pragma: no cover
    stdio_client = None
    ClientSession = None

try:
    from interpreter import interpreter  # type: ignore
except Exception:  # pragma: no cover
    interpreter = None


# ---------------------------------------------------------
# 로깅 설정
# ---------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mega_ai_bot")


# ---------------------------------------------------------
# 모델 프로필 및 설정
# ---------------------------------------------------------
@dataclass(frozen=True)
class ModelProfile:
    name: str
    # Ollama keep_alive:
    # -1: 모델 상주, 0: 즉시 해제, 또는 "5m" 같은 duration 문자열
    keep_alive: Any
    temperature: float = 0.2


@dataclass
class Settings:
    telegram_bot_token: str
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_sec: float = 120.0
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "conversation_memory"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    planner: ModelProfile = field(
        default_factory=lambda: ModelProfile("llama3.1:8b", keep_alive=-1, temperature=0.1)
    )
    specialist_code: ModelProfile = field(
        default_factory=lambda: ModelProfile("qwen3-coder:30b", keep_alive=0, temperature=0.2)
    )
    specialist_reason: ModelProfile = field(
        default_factory=lambda: ModelProfile("deepseek-r1:32b", keep_alive=0, temperature=0.2)
    )
    verifier: ModelProfile = field(
        default_factory=lambda: ModelProfile("qwen3.5:35b", keep_alive=0, temperature=0.1)
    )
    synthesizer: ModelProfile = field(
        default_factory=lambda: ModelProfile("gemma2:9b", keep_alive=-1, temperature=0.3)
    )

    # MCP 서버: JSON 문자열 예시
    # [{"name":"filesystem","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","."]}]
    mcp_servers_json: str = "[]"


def load_settings() -> Settings:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN 이 설정되지 않았습니다.")

    return Settings(
        telegram_bot_token=token,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_timeout_sec=float(os.getenv("OLLAMA_TIMEOUT_SEC", "120")),
        chroma_path=os.getenv("CHROMA_PATH", "./chroma_db"),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "conversation_memory"),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        ),
        mcp_servers_json=os.getenv("MCP_SERVERS_JSON", "[]"),
    )


# ---------------------------------------------------------
# Ollama 비동기 클라이언트
# ---------------------------------------------------------
class OllamaClient:
    def __init__(self, base_url: str, timeout_sec: float) -> None:
        self.base_url = base_url
        self.timeout = httpx.Timeout(timeout_sec)
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self) -> None:
        await self.client.aclose()

    async def chat(
        self,
        model: ModelProfile,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": model.name,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": model.temperature},
            "keep_alive": self._normalize_keep_alive(model.keep_alive),
        }
        if tools:
            payload["tools"] = tools

        try:
            resp = await self.client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data
        except httpx.TimeoutException as e:
            logger.exception("Ollama 타임아웃 발생")
            raise RuntimeError(f"Ollama timeout: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.exception("Ollama HTTP 오류")
            raise RuntimeError(f"Ollama bad response: {e.response.text}") from e
        except Exception as e:
            logger.exception("Ollama 호출 중 알 수 없는 오류")
            raise RuntimeError(f"Ollama unknown error: {e}") from e

    @staticmethod
    def _normalize_keep_alive(keep_alive: Any) -> Any:
        """
        Ollama 버전에 따라 keep_alive가 duration 문자열(예: \"10m\") 또는 정수(-1/0)로 처리됩니다.
        과거 설정값(\"-1\", \"0\")이 문자열로 들어오면 정수로 변환해 400 에러를 방지합니다.
        """
        if isinstance(keep_alive, str) and keep_alive.strip() in {"-1", "0"}:
            return int(keep_alive.strip())
        return keep_alive


# ---------------------------------------------------------
# ChromaDB 기반 장기 기억(RAG)
# ---------------------------------------------------------
class MemoryStore:
    def __init__(self, persist_path: str, collection_name: str, embedding_model: str) -> None:
        emb_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"},
        )

    async def add_turn(self, user_id: int, user_text: str, assistant_text: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        doc = (
            f"[time] {timestamp}\n"
            f"[user_id] {user_id}\n"
            f"[user] {user_text}\n"
            f"[assistant] {assistant_text}"
        )
        metadata = {
            "user_id": str(user_id),
            "timestamp": timestamp,
            "type": "conversation_turn",
        }
        uid = str(uuid.uuid4())
        await asyncio.to_thread(
            self.collection.add,
            ids=[uid],
            documents=[doc],
            metadatas=[metadata],
        )

    async def retrieve_similar(self, user_id: int, query: str, k: int = 5) -> List[str]:
        try:
            result = await asyncio.to_thread(
                self.collection.query,
                query_texts=[query],
                n_results=k,
                where={"user_id": str(user_id)},
            )
            docs = result.get("documents", [[]])[0]
            return docs if docs else []
        except Exception:
            # 사용자별 필터가 없는 기존 데이터/예외 시 전체 검색 fallback
            logger.warning("사용자 필터 검색 실패 -> 전체 검색 fallback", exc_info=True)
            try:
                result = await asyncio.to_thread(
                    self.collection.query,
                    query_texts=[query],
                    n_results=k,
                )
                docs = result.get("documents", [[]])[0]
                return docs if docs else []
            except Exception:
                logger.exception("Chroma 검색 최종 실패")
                return []


# ---------------------------------------------------------
# 도구(Tool) 인터페이스 및 구현
# ---------------------------------------------------------
class ToolRuntimeError(Exception):
    pass


class ToolBase:
    name: str
    description: str
    input_schema: Dict[str, Any]

    def as_ollama_schema(self) -> Dict[str, Any]:
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
    description = "파이썬/셸 작업을 Open Interpreter로 수행합니다."
    input_schema = {
        "type": "object",
        "properties": {
            "instruction": {"type": "string", "description": "실행 지시문"},
        },
        "required": ["instruction"],
    }

    async def run(self, arguments: Dict[str, Any]) -> str:
        if interpreter is None:
            raise ToolRuntimeError("Open Interpreter를 import할 수 없습니다.")
        instruction = arguments.get("instruction", "")
        if not instruction:
            raise ToolRuntimeError("instruction 인자가 비어 있습니다.")

        def _run_sync() -> str:
            try:
                result = interpreter.chat(instruction)
                return json.dumps(result, ensure_ascii=False)
            except Exception as e:
                raise ToolRuntimeError(f"Open Interpreter 실행 실패: {e}") from e

        return await asyncio.to_thread(_run_sync)


class MCPTool(ToolBase):
    def __init__(self, server_name: str, tool_name: str, description: str, input_schema: Dict[str, Any], call_fn):
        self.server_name = server_name
        self.name = f"mcp::{server_name}::{tool_name}"
        self.description = description or f"MCP tool {tool_name}"
        self.input_schema = input_schema or {"type": "object", "properties": {}}
        self._call_fn = call_fn

    async def run(self, arguments: Dict[str, Any]) -> str:
        try:
            result = await self._call_fn(arguments)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            raise ToolRuntimeError(f"MCP tool 실행 실패({self.name}): {e}") from e


class MCPManager:
    """
    MCP 서버와 연결하고 사용 가능한 도구를 동적으로 적재합니다.
    공식 SDK의 버전 차이를 고려하여 실패 시 graceful fallback 합니다.
    """

    def __init__(self, server_configs: List[Dict[str, Any]]) -> None:
        self.server_configs = server_configs
        self.tools: List[MCPTool] = []
        self._sessions: List[Any] = []
        self._stdio_contexts: List[Any] = []

    async def connect_and_discover(self) -> List[MCPTool]:
        if not self.server_configs:
            logger.info("MCP 서버 설정이 없어 건너뜁니다.")
            return []
        if stdio_client is None or ClientSession is None:
            logger.warning("MCP SDK import 실패로 MCP 기능 비활성화")
            return []

        for cfg in self.server_configs:
            name = cfg.get("name", "unknown")
            command = cfg.get("command")
            args = cfg.get("args", [])
            env = cfg.get("env", None)
            if not command:
                logger.warning("MCP 서버(%s) command 누락", name)
                continue

            try:
                stdio_cm = stdio_client(command=command, args=args, env=env)
                read_stream, write_stream = await stdio_cm.__aenter__()
                session = ClientSession(read_stream, write_stream)
                await session.__aenter__()
                await session.initialize()

                tools_result = await session.list_tools()
                for t in tools_result.tools:
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

                self._stdio_contexts.append(stdio_cm)
                self._sessions.append(session)
                logger.info("MCP 서버 연결 성공: %s (tools=%d)", name, len(tools_result.tools))
            except Exception:
                logger.exception("MCP 서버 연결 실패: %s", name)

        return self.tools

    async def close(self) -> None:
        for session in reversed(self._sessions):
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                logger.warning("MCP session 종료 실패", exc_info=True)
        for stdio_cm in reversed(self._stdio_contexts):
            try:
                await stdio_cm.__aexit__(None, None, None)
            except Exception:
                logger.warning("MCP stdio 종료 실패", exc_info=True)


# ---------------------------------------------------------
# 멀티 에이전트 오케스트레이터
# ---------------------------------------------------------
@dataclass
class PipelineResult:
    plan_text: str
    specialist_draft: str
    verifier_feedback: str
    final_answer: str


class MultiAgentOrchestrator:
    def __init__(self, settings: Settings, ollama: OllamaClient, memory: MemoryStore, tools: List[ToolBase]) -> None:
        self.settings = settings
        self.ollama = ollama
        self.memory = memory
        self.tools = {t.name: t for t in tools}

    async def _run_planner_with_tools(self, user_id: int, user_query: str, mode: str) -> Tuple[str, List[Dict[str, Any]]]:
        memories = await self.memory.retrieve_similar(user_id=user_id, query=user_query, k=5)
        memory_context = "\n\n".join(memories) if memories else "(유사 과거 대화 없음)"

        tool_schemas = [t.as_ollama_schema() for t in self.tools.values()]

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "당신은 Planner 에이전트입니다.\n"
                    "- 사용자 요청을 분석하고, 필요시 tool_calls로 도구를 호출하세요.\n"
                    "- 도구 실행이 끝나면 최종적으로 한국어 계획서(단계별)를 작성하세요.\n"
                    "- 모드: {mode}\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"[현재 사용자 질문]\n{user_query}\n\n"
                    f"[유사 과거 대화 Top-5]\n{memory_context}\n"
                ),
            },
        ]

        tool_trace: List[Dict[str, Any]] = []

        for _ in range(6):  # 무한 루프 방지
            planner_resp = await self.ollama.chat(
                model=self.settings.planner,
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
            )
            message = planner_resp.get("message", {})
            tool_calls = message.get("tool_calls", []) or []

            if not tool_calls:
                plan_text = message.get("content", "계획 생성 실패")
                return plan_text, tool_trace

            messages.append({"role": "assistant", "content": message.get("content", ""), "tool_calls": tool_calls})

            for tc in tool_calls:
                function_info = tc.get("function", {})
                tool_name = function_info.get("name")
                raw_args = function_info.get("arguments", {})
                args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")

                if tool_name not in self.tools:
                    result = f"[오류] 존재하지 않는 도구: {tool_name}"
                else:
                    try:
                        result = await self.tools[tool_name].run(args)
                    except Exception as e:
                        result = f"[도구 실행 실패] {e}"

                tool_trace.append({"tool": tool_name, "arguments": args, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": result,
                    }
                )

        return "계획 생성 중 도구 루프 한도 초과", tool_trace

    async def _run_specialist(self, mode: str, user_query: str, plan_text: str, tool_trace: List[Dict[str, Any]]) -> str:
        specialist_model = self.settings.specialist_code if mode == "code" else self.settings.specialist_reason
        prompt = (
            "당신은 Specialist 에이전트입니다.\n"
            f"모드: {mode}\n"
            "아래 계획/도구 결과를 바탕으로 초안을 작성하세요.\n\n"
            f"[사용자 요청]\n{user_query}\n\n"
            f"[계획]\n{plan_text}\n\n"
            f"[도구 실행 로그]\n{json.dumps(tool_trace, ensure_ascii=False, indent=2)}"
        )
        resp = await self.ollama.chat(
            model=specialist_model,
            messages=[{"role": "user", "content": prompt}],
            tools=None,
        )
        return resp.get("message", {}).get("content", "초안 생성 실패")

    async def _run_verifier(self, user_query: str, specialist_draft: str) -> str:
        prompt = (
            "당신은 Verifier 에이전트입니다.\n"
            "아래 초안의 논리적 오류/코드 버그/누락 사항을 비판적으로 검토하세요.\n"
            "출력 형식: 문제점, 근거, 수정 제안\n\n"
            f"[사용자 요청]\n{user_query}\n\n"
            f"[초안]\n{specialist_draft}"
        )
        resp = await self.ollama.chat(
            model=self.settings.verifier,
            messages=[{"role": "user", "content": prompt}],
            tools=None,
        )
        return resp.get("message", {}).get("content", "검토 실패")

    async def _run_synthesizer(self, user_query: str, plan_text: str, specialist_draft: str, verifier_feedback: str) -> str:
        prompt = (
            "당신은 Synthesizer 에이전트입니다.\n"
            "최종 답변을 한국어 Markdown으로 사용자 친화적으로 작성하세요.\n"
            "필요하면 코드 블록과 체크리스트를 활용하세요.\n\n"
            f"[사용자 요청]\n{user_query}\n\n"
            f"[계획]\n{plan_text}\n\n"
            f"[초안]\n{specialist_draft}\n\n"
            f"[검토]\n{verifier_feedback}"
        )
        resp = await self.ollama.chat(
            model=self.settings.synthesizer,
            messages=[{"role": "user", "content": prompt}],
            tools=None,
        )
        return resp.get("message", {}).get("content", "최종 합성 실패")

    async def run(self, user_id: int, mode: str, user_query: str) -> PipelineResult:
        plan_text, tool_trace = await self._run_planner_with_tools(user_id=user_id, user_query=user_query, mode=mode)
        specialist_draft = await self._run_specialist(
            mode=mode,
            user_query=user_query,
            plan_text=plan_text,
            tool_trace=tool_trace,
        )
        verifier_feedback = await self._run_verifier(user_query=user_query, specialist_draft=specialist_draft)
        final_answer = await self._run_synthesizer(
            user_query=user_query,
            plan_text=plan_text,
            specialist_draft=specialist_draft,
            verifier_feedback=verifier_feedback,
        )
        return PipelineResult(
            plan_text=plan_text,
            specialist_draft=specialist_draft,
            verifier_feedback=verifier_feedback,
            final_answer=final_answer,
        )


# ---------------------------------------------------------
# Telegram Bot 애플리케이션
# ---------------------------------------------------------
class TelegramAIAssistantApp:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bot = Bot(token=settings.telegram_bot_token)
        self.dp = Dispatcher()
        self.ollama = OllamaClient(settings.ollama_base_url, settings.ollama_timeout_sec)
        self.memory = MemoryStore(
            persist_path=settings.chroma_path,
            collection_name=settings.chroma_collection,
            embedding_model=settings.embedding_model_name,
        )

        self.mcp_manager = MCPManager(server_configs=self._parse_mcp_servers(settings.mcp_servers_json))
        self.orchestrator: Optional[MultiAgentOrchestrator] = None

        self.dp.message.register(self.on_start, Command("start"))
        self.dp.message.register(self.on_help, Command("help"))
        self.dp.message.register(self.on_code, Command("code"))
        self.dp.message.register(self.on_reason, Command("reason"))
        self.dp.message.register(self.on_default, F.text)

    @staticmethod
    def _parse_mcp_servers(raw: str) -> List[Dict[str, Any]]:
        # dotenv는 멀티라인 JSON 파싱에 취약할 수 있어, 먼저 정규 JSON 문자열을 기대합니다.
        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except Exception:
            logger.warning("MCP_SERVERS_JSON 파싱 실패 (JSON 배열 문자열인지 확인 필요)")
            return []

    async def setup(self) -> None:
        tools: List[ToolBase] = [OpenInterpreterTool()]
        mcp_tools = await self.mcp_manager.connect_and_discover()
        tools.extend(mcp_tools)

        self.orchestrator = MultiAgentOrchestrator(
            settings=self.settings,
            ollama=self.ollama,
            memory=self.memory,
            tools=tools,
        )
        logger.info("앱 초기화 완료 (tools=%d)", len(tools))

    async def shutdown(self) -> None:
        await self.mcp_manager.close()
        await self.ollama.close()
        await self.bot.session.close()

    async def on_start(self, message: Message) -> None:
        await message.answer(
            "안녕하세요! 멀티 에이전트 로컬 AI 비서입니다.\n"
            "- /code <질문>: 코딩 중심 분석\n"
            "- /reason <질문>: 추론 중심 분석\n"
            "일반 텍스트는 /reason 모드로 처리합니다."
        )

    async def on_help(self, message: Message) -> None:
        await self.on_start(message)

    async def on_code(self, message: Message) -> None:
        text = (message.text or "").replace("/code", "", 1).strip()
        if not text:
            await message.answer("사용법: /code FastAPI 인증 미들웨어 예시를 만들어줘")
            return
        await self._handle_user_query(message, mode="code", query=text)

    async def on_reason(self, message: Message) -> None:
        text = (message.text or "").replace("/reason", "", 1).strip()
        if not text:
            await message.answer("사용법: /reason 벡터DB와 RDB를 언제 같이 써야 해?")
            return
        await self._handle_user_query(message, mode="reason", query=text)

    async def on_default(self, message: Message) -> None:
        text = (message.text or "").strip()
        if not text:
            return
        await self._handle_user_query(message, mode="reason", query=text)

    async def _handle_user_query(self, message: Message, mode: str, query: str) -> None:
        if self.orchestrator is None:
            await message.answer("초기화 중입니다. 잠시 후 다시 시도해주세요.")
            return

        # 무거운 추론 중에도 이벤트 루프를 막지 않기 위해 task로 분리
        asyncio.create_task(self._process_and_reply(message, mode, query))
        await message.answer("요청을 처리 중입니다... (멀티 에이전트 파이프라인 실행)")

    async def _process_and_reply(self, message: Message, mode: str, query: str) -> None:
        assert self.orchestrator is not None
        try:
            result = await self.orchestrator.run(user_id=message.from_user.id, mode=mode, user_query=query)
            await message.answer(result.final_answer)
            await self.memory.add_turn(
                user_id=message.from_user.id,
                user_text=query,
                assistant_text=result.final_answer,
            )
        except RuntimeError as e:
            logger.error("런타임 오류: %s", e)
            await message.answer(
                "처리 중 오류가 발생했습니다.\n"
                f"- 원인: {e}\n"
                "잠시 후 다시 시도하거나 질문을 더 짧게 나눠서 보내주세요."
            )
        except Exception as e:
            logger.exception("예상치 못한 오류")
            await message.answer(
                "예상치 못한 내부 오류가 발생했습니다.\n"
                f"- 오류: {e}\n"
                "로그를 확인해주세요."
            )
            logger.debug(traceback.format_exc())

    async def run(self) -> None:
        await self.setup()
        try:
            await self.dp.start_polling(self.bot)
        finally:
            await self.shutdown()


async def main() -> None:
    settings = load_settings()
    app = TelegramAIAssistantApp(settings)
    await app.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("사용자 중단으로 종료합니다.")
