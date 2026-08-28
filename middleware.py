"""
의료 정보 제공 및 안내 에이전트 - 커스텀 미들웨어

이 파일은 Day7 도구 실습 이후, LangGraph Studio 연동 및 Claude Code
활성화 단계에서 구현되었습니다.

구현된 미들웨어:
1. emergency_screening_middleware (@before_agent)
   - 사용자의 최신 메시지에서 응급 키워드를 선제적으로 감지하여,
     LLM이 도구 호출 여부를 판단하기 전에 응급 안내를 최우선 지침으로 주입합니다.
2. disclaimer_guard_middleware (@wrap_tool_call)
   - 커스텀 도구 실행 결과에 "참고용" 안내 문구가 누락된 경우 자동으로 보강합니다.
3. tool_call_logging_middleware (@wrap_tool_call)
   - 모든 도구 호출을 로컬 로그 파일(logs/tool_calls.log)에 기록하여
     추후 감사(audit)나 오남용 점검에 활용할 수 있도록 합니다.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage
from langchain.agents.middleware import before_agent, wrap_tool_call, AgentState
from langgraph.runtime import Runtime

# tools.py와 동일한 기준을 사용합니다.
from tools import EMERGENCY_KEYWORDS, DISCLAIMER, SYMPTOM_KEYWORDS, load_skill

# 자동 주입되는 skill을 식별하기 위한 마커 (중복 주입 방지용)
SKILL_MARKER = "[SKILL:symptom-navigation]"

# disclaimer 보강이 필요한 커스텀 도구 (참고 정보를 반환하는 도구들)
DISCLAIMER_REQUIRED_TOOLS = {
    "search_symptom_info",
    "lookup_drug_info",
    "get_health_info",
}

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "tool_calls.log"


def _check_emergency(text: str) -> bool:
    return any(keyword in text for keyword in EMERGENCY_KEYWORDS)


def _check_symptom(text: str) -> bool:
    return any(keyword in text for keyword in SYMPTOM_KEYWORDS)


def _skill_already_injected(state: AgentState) -> bool:
    """이번 대화에서 symptom-navigation skill이 이미 주입되었는지 확인합니다."""
    messages = state.get("messages", [])
    for message in messages:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content", "")
        if isinstance(content, str) and SKILL_MARKER in content:
            return True
    return False


def _latest_user_text(state: AgentState) -> str:
    """state의 메시지 목록에서 가장 최근 사용자(human) 메시지의 텍스트를 추출합니다."""
    messages = state.get("messages", [])
    for message in reversed(messages):
        # LangChain 메시지 객체 또는 dict 형태 모두 지원
        msg_type = getattr(message, "type", None) or (
            message.get("type") if isinstance(message, dict) else None
        )
        if msg_type == "human":
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


@before_agent
def emergency_screening_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Emergency Screening Middleware

    에이전트가 도구를 호출하기 전에, 사용자의 최신 메시지에 응급 키워드가
    포함되어 있는지 먼저 검사합니다. 감지되면 LLM에게 검색 도구 호출보다
    응급 안내(119/응급실)를 최우선으로 답변하도록 지시하는 SystemMessage를 주입합니다.
    """
    user_text = _latest_user_text(state)

    if not user_text or not _check_emergency(user_text):
        return None

    print(f"\n[Emergency Screening] 🚨 응급 키워드 감지: {user_text[:50]}")

    system_message = SystemMessage(
        content=(
            "[응급 감지 알림]\n"
            "사용자의 메시지에서 응급 가능성이 있는 증상 키워드가 감지되었습니다.\n"
            "도구를 호출하기 전에, 먼저 사용자에게 즉시 119에 신고하거나 "
            "가까운 응급실을 방문하도록 안내하세요. "
            "이 경우 증상 검색 도구 호출은 생략해도 됩니다."
        )
    )

    return {"messages": [system_message]}


@before_agent
def skill_injection_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Skill Injection Middleware

    사용자가 신체 증상을 언급하면(응급 상황은 제외), '증상 입력 → 응급 여부 판단 →
    진료과 추천 → 병원 찾기 → 오늘 진료 여부 확인 → 병원 방문 체크리스트'로 이어지는
    절차를 기술한 skills/symptom-navigation/skill.md를 SystemMessage로 자동 주입합니다.

    - emergency_screening_middleware보다 뒤에 실행되도록 agent.py의 미들웨어 목록
      순서를 [emergency_screening_middleware, skill_injection_middleware, ...] 로 둡니다.
    - 응급 키워드가 감지된 경우에는 skill을 주입하지 않습니다 (응급 안내가 최우선).
    - 동일 대화 내에서 한 번만 주입되도록 SKILL_MARKER로 중복 주입을 방지합니다.
    - LLM의 tool-call 판단을 거치지 않고 미들웨어 단계에서 자동으로 주입되는 방식입니다.
    """
    if _skill_already_injected(state):
        return None

    user_text = _latest_user_text(state)
    if not user_text:
        return None

    # 응급 상황이면 절차 전체보다 응급 안내가 우선이므로 주입을 건너뜁니다.
    if _check_emergency(user_text):
        return None

    if not _check_symptom(user_text):
        return None

    skill_content = load_skill()
    if not skill_content:
        print("[Skill Injection] ⚠️ skill.md 로드 실패 - 파일을 찾을 수 없습니다.")
        return None

    print(f"[Skill Injection] 📋 증상 입력 감지 - symptom-navigation skill 주입: {user_text[:50]}")

    system_message = SystemMessage(content=f"{SKILL_MARKER}\n{skill_content}")

    return {"messages": [system_message]}


@wrap_tool_call
async def disclaimer_guard_middleware(request, handler):
    """Disclaimer Guard Middleware

    search_symptom_info, lookup_drug_info, get_health_info 등
    참고 정보를 제공하는 도구의 실행 결과에 "진단·처방 대체 불가" 안내 문구가
    누락되어 있으면 자동으로 보강합니다. (LLM 또는 도구 수정 실수로 인한
    안내 누락을 방지하는 이중 안전장치)
    """
    tool_name = request.tool_call["name"]

    result = await handler(request)

    if tool_name not in DISCLAIMER_REQUIRED_TOOLS:
        return result

    try:
        content = getattr(result, "content", None)
        if isinstance(content, str) and "참고용" not in content and "응급" not in content:
            result.content = content + DISCLAIMER
            print(f"[Disclaimer Guard] ℹ️ '{tool_name}' 결과에 안내 문구 보강")
    except Exception as e:
        print(f"[Disclaimer Guard] ⚠️ 안내 문구 보강 실패: {e}")

    return result


@wrap_tool_call
async def tool_call_logging_middleware(request, handler):
    """Tool Call Logging Middleware

    모든 도구 호출(도구명, 입력값, 호출 시각)을 logs/tool_calls.log 파일에 기록합니다.
    의료 정보를 다루는 에이전트 특성상, 어떤 요청에 어떤 도구가 사용되었는지
    감사(audit) 목적으로 추적할 수 있도록 합니다.
    """
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = await handler(request)

    try:
        LOG_DIR.mkdir(exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] tool={tool_name} args={tool_args}\n")
    except Exception as e:
        print(f"[Tool Call Logging] ⚠️ 로그 기록 실패: {e}")

    return result
