# Mega Local Multi-Agent AI Assistant Bot

Telegram + Ollama(로컬 LLM) + MCP + Open Interpreter + ChromaDB를 통합한 **비동기 멀티 에이전트 로컬 AI 비서**입니다.

## 1) 주요 기능

- **완전 비동기 Telegram Bot** (`aiogram`, `asyncio`)
  - 무거운 추론 작업 중에도 다른 메시지를 계속 수신하도록 설계.
- **역할 분리 멀티 에이전트 파이프라인**
  1. Planner (`llama3.1:8b`)
  2. Specialist (`/code` → `qwen3-coder:30b`, `/reason` → `deepseek-r1:32b`)
  3. Verifier (`qwen3.5:35b`)
  4. Synthesizer (`gemma2:9b`)
- **RAG 기반 장기 기억** (`chromadb`)
  - 사용자 대화를 임베딩 저장 후 유사 대화 Top-5를 Planner 문맥에 주입.
- **Tool Calling 루프**
  - MCP 서버 도구(filesystem/sqlite 등) 자동 탐색 + Open Interpreter 래핑 도구 실행.

---

## 2) 아키텍처 개요

```
Telegram Message
   ↓
TelegramAIAssistantApp (aiogram handlers)
   ↓
MultiAgentOrchestrator.run()
   ├─ Planner (tool_calls + MCP/Open Interpreter 실행 루프)
   ├─ Specialist (mode에 따라 code/reason 모델 선택)
   ├─ Verifier (초안 비판적 검토)
   └─ Synthesizer (한국어 Markdown 최종 답변)
   ↓
MemoryStore.add_turn() -> ChromaDB 저장
```

---

## 3) 프로젝트 구조

- `bot.py`: 전체 애플리케이션 코드
  - 설정 로딩
  - Ollama 비동기 클라이언트
  - ChromaDB 메모리 저장소
  - MCP/Open Interpreter 도구 계층
  - 멀티 에이전트 오케스트레이터
  - Telegram 이벤트 핸들러
- `requirements.txt`: 파이썬 패키지 목록
- `.env.example`: 실행 환경 변수 예시

---

## 4) 사전 요구사항

- Python 3.11+
- Ollama 설치 및 모델 pull
- Telegram Bot Token
- (선택) Node.js + npx (MCP 서버 stdio 실행 시)

### Ollama 모델 준비 예시

```bash
ollama pull llama3.1:8b
ollama pull qwen3-coder:30b
ollama pull deepseek-r1:32b
ollama pull qwen3.5:35b
ollama pull gemma2:9b
```

> GPU VRAM 여유가 부족하면 Specialist/Verifier 모델 호출 시 지연이 발생할 수 있습니다.

---

## 5) 설치 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

환경 변수 파일 생성:

```bash
cp .env.example .env
```

`.env`에서 최소 아래 항목을 반드시 설정하세요.

- `TELEGRAM_BOT_TOKEN`
- `OLLAMA_BASE_URL`
- `MCP_SERVERS_JSON` (필요 시)

---

## 6) 실행 방법

```bash
python bot.py
```

정상 실행 시 Telegram에서:

- `/start` : 안내 메시지
- `/help` : 안내 메시지
- `/code <질문>` : 코드 중심 Specialist 사용
- `/reason <질문>` : 추론 중심 Specialist 사용
- 일반 텍스트 : 기본적으로 `/reason` 모드로 처리

---

## 7) 설정 가이드 (.env)

### 필수

- `TELEGRAM_BOT_TOKEN`: BotFather에서 발급받은 토큰

### Ollama

- `OLLAMA_BASE_URL`: 기본 `http://localhost:11434`
- `OLLAMA_TIMEOUT_SEC`: 모델 응답 타임아웃(초)

### ChromaDB

- `CHROMA_PATH`: 벡터 저장 경로
- `CHROMA_COLLECTION`: 컬렉션 이름
- `EMBEDDING_MODEL_NAME`: 임베딩 모델명

### MCP

- `MCP_SERVERS_JSON`: MCP 서버 목록(JSON 배열)

예시:

```json
[
  {
    "name": "filesystem",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
  },
  {
    "name": "sqlite",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "./data.db"]
  }
]
```

---

## 8) 운영/개발 가이드

### 8-1. 로깅

- `LOG_LEVEL=INFO`(기본), 필요 시 `DEBUG`로 조정
- Ollama timeout/HTTP 오류, MCP 연결 실패 시 fallback 로그 출력

### 8-2. 성능 팁

- Planner/Synthesizer는 `keep_alive=-1`로 상주
- Specialist/Verifier는 `keep_alive=0`으로 작업 후 해제
- 대규모 모델 사용 시 동시 요청 폭주를 막기 위해 큐/세마포어 추가를 권장

### 8-3. 장애 대응

- MCP SDK import 실패 시: MCP 기능 비활성화 상태로 계속 동작
- Open Interpreter import 실패 시: 해당 도구만 비활성화
- Chroma 조회 실패 시: 사용자 필터 해제 후 재조회 fallback

---

## 9) 보안 주의사항

- `TELEGRAM_BOT_TOKEN`은 절대 커밋하지 마세요.
- Open Interpreter/MCP 도구는 로컬 파일/명령 실행 권한과 연결되므로 신뢰 가능한 입력만 허용하세요.
- 운영 환경에서는 사용자별 접근 제어, 실행 제한, 감사 로깅을 추가하세요.

---

## 10) 빠른 점검 체크리스트

- [ ] `ollama serve` 실행 중인가?
- [ ] 필요한 모델이 pull 되어 있는가?
- [ ] `.env`의 Telegram 토큰이 유효한가?
- [ ] (선택) `MCP_SERVERS_JSON` 명령이 로컬에서 실행 가능한가?
- [ ] `python bot.py` 실행 시 초기화 로그가 정상 출력되는가?

