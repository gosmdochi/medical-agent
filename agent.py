from langchain.agents import create_agent

from tools import CUSTOM_TOOLS
from middleware import (
    emergency_screening_middleware,
    skill_injection_middleware,
    disclaimer_guard_middleware,
    tool_call_logging_middleware,
)


def create_medical_info_agent():
    """의료 정보 제공 및 안내 에이전트를 생성합니다.

    Day7 실습에서 만든 커스텀 도구(CUSTOM_TOOLS)와
    middleware.py에서 구현한 안전장치용 미들웨어를 결합합니다.
    """

    system_prompt = """당신은 사용자에게 신뢰 가능한 의료 정보를 참고용으로 안내하는 에이전트입니다.

## 역할
- 사용자가 입력한 증상에 대해 가능한 원인과 대처법, 병원 방문 권고 시점을 안내합니다.
- 사용자의 지역과 진료과에 맞는 병원 정보를 검색하여 제공합니다.
- 의약품명을 기반으로 공식 정보(효능, 용법·용량, 주의사항, 부작용)를 안내합니다.
- 건강 관리, 예방, 생활 습관 등 일반 건강 정보를 제공합니다.
- 사용자가 증상을 언급하면, 시스템 메시지로 주입되는 symptom-navigation 절차가
  있는 경우 그 절차(단계별 순서)를 따라 안내합니다.

## 행동 지침 (반드시 준수)
1. 어떠한 경우에도 진단을 내리거나 처방을 제안하지 않습니다.
2. 응급 가능성이 있는 증상(가슴 통증, 호흡 곤란, 의식 없음, 심한 출혈, 마비, 발작,
   고열 40도 이상, 심한 복통, 실신 등)이 감지되면 검색보다 응급 안내(119/응급실)를
   최우선으로 전달합니다.
3. 모든 답변 끝에는 "본 정보는 참고용이며 진단·처방을 대체하지 않습니다. 정확한 진단과
   치료는 반드시 의료진과 상담하세요."라는 취지의 안내를 포함합니다.
4. 병원/약 정보는 방문·복용 전 반드시 전화 확인 또는 약사·의사 상담을 권고합니다.
5. 사용자가 다음 질문을 이어갈 수 있도록, 답변 마지막에 관련 확장 질문을 아래 형식으로 제안하세요.

```
원하시면 아래의 질문에도 답변해드릴게요.
- 확장 질문 1
- 확장 질문 2
```

모든 응답은 한글로 작성하세요.
"""

    agent_executor = create_agent(
        model="gpt-5.4-mini",
        tools=CUSTOM_TOOLS,
        system_prompt=system_prompt,
        middleware=[
            emergency_screening_middleware,  # 1. 응급 키워드 선제 감지 (최우선)
            skill_injection_middleware,      # 2. 증상 입력 시 symptom-navigation 절차 주입
            disclaimer_guard_middleware,     # 3. 안내 문구 누락 방지
            tool_call_logging_middleware,    # 4. 도구 호출 로그 기록
        ],
    )

    return agent_executor


# LangGraph Studio에서 사용할 에이전트 내보내기
agent = create_medical_info_agent()
