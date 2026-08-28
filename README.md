# 의료 정보 제공 및 안내 에이전트 (Medical Info Agent)

Day7 실습(`day7_team_project_template.ipynb`)에서 설계·생성한 커스텀 도구를
LangGraph 기반 에이전트에 통합하고, `skills/symptom-navigation/skill.md` 절차와
안전장치 미들웨어를 결합한 팀 프로젝트 저장소입니다.

> ⚠️ 본 에이전트는 의료 "참고 정보"만 제공하며, 진단이나 처방을 대체하지 않습니다.
> 모든 응답에는 전문의 상담을 권고하는 안내 문구가 포함됩니다.

---

## 팀 도메인

**의료 정보 제공 및 안내 에이전트**

사용자의 증상, 위치, 약품명, 건강 주제를 입력받아 신뢰 가능한 출처(질병관리청, 대형병원,
식약처 등)에서 검색한 참고 정보를 제공합니다. 응급 가능성이 있는 증상이 감지되면
검색 없이 119/응급실 안내를 최우선으로 제공합니다. 일반 증상이 감지되면
`symptom-navigation` skill 절차(증상 확인 → 진료과 추천 → 병원 찾기 → 진료 여부 확인 →
방문 체크리스트)를 따라 단계별로 안내합니다.
<img width="1136" height="608" alt="스크린샷 2026-08-28 133819" src="https://github.com/user-attachments/assets/5e707ead-4f59-4bd2-a697-1b0cd9728bb4" />

---

## 파일 구조

```
medical-info-agent/
├── day7_team_project_template.ipynb        # 도구 설계 및 단독 테스트 (실습 기록)
├── agent.py                                  # 에이전트 정의 (시스템 프롬프트 + 도구 + 미들웨어)
├── tools.py                                  # 커스텀 도구 (CUSTOM_TOOLS) + skill.md 로더(load_skill)
├── middleware.py                             # 커스텀 미들웨어 (Claude Code로 구현)
├── skills/
│   └── symptom-navigation/
│       └── skill.md                          # 증상 안내 절차 skill
├── langgraph.json                            # LangGraph Studio 실행 설정
├── .env                                      # 환경 변수 (Git 업로드 X)
├── .env.example                              # 환경 변수 예시
├── .gitignore
├── pyproject.toml                            # 프로젝트 의존성 정의
├── uv.lock                                   # 의존성 잠금 파일
└── README.md
```

---

## 도구 목록 (`tools.py` → `CUSTOM_TOOLS`)

| 도구명 | 입력 | 설명 |
|---|---|---|
| `search_symptom_info` | `symptom` (증상) | 신뢰 가능한 의학 정보 출처에서 원인/대처법 검색. 응급 키워드 감지 시 검색 없이 응급 안내 반환 |
| `find_hospital` | `location`, `department`(선택) | 지역/진료과 기준 병원 정보 검색 |
| `lookup_drug_info` | `drug_name` (약품명) | 식약처 등 공식 DB에서 효능·용법·주의사항·부작용 조회 |
| `get_health_info` | `topic` (건강 주제) | 예방/생활 습관 등 일반 건강 정보 검색 |
| `recommend_department` | `symptom` (증상) | 증상 키워드를 기반으로 적절한 진료과 1~2개 추천 (`symptom-navigation` skill 3단계에서 사용) |
| `check_hospital_hours` | `hospital_name` (병원명) | 병원의 오늘 진료 여부 확인 (현재 TODO: 외부 API 미연동, 임시 응답) |
| `generate_visit_checklist` | `symptom`, `department` | 병원 방문 전 준비물/확인사항 체크리스트 생성 |

모든 검색형 도구는 `TavilySearch`를 `include_domains`로 제한하여 신뢰 가능한 출처만 검색하며,
진단·처방 표현을 금지하고 참고 정보 제공에 한정합니다.

`load_skill(skill_name="symptom-navigation")` 함수는 `skills/<skill_name>/skill.md`를
읽어 문자열로 반환하며, LLM의 도구 호출 판단을 거치지 않고 `middleware.py`의
`skill_injection_middleware`가 직접 호출합니다.

---

## Skill: `symptom-navigation` (`skills/symptom-navigation/skill.md`)

사용자가 신체 증상을 언급하면(응급 상황 제외), 아래 6단계 절차를 순서대로 안내합니다.

1. 증상 입력 수집
2. 응급 여부 판단 (응급이면 절차 중단 → 119/응급실 안내로 즉시 전환)
3. 진료과 추천 (`recommend_department`)
4. 병원 찾기 (`find_hospital`)
5. 오늘 진료 여부 확인 (`check_hospital_hours`)
6. 병원 방문 체크리스트 제공 (`generate_visit_checklist`)

이 절차는 LLM이 자율적으로 선택하는 것이 아니라, `skill_injection_middleware`가
사용자 메시지에서 일반 증상 키워드(`SYMPTOM_KEYWORDS`)를 감지했을 때 시스템 메시지로
자동 주입합니다.

---

## 미들웨어 동작 방식 (`middleware.py`)

Day7에서 만든 도구를 그대로 쓰지 않고, Claude Code를 활성화한 이후
아래 4가지 안전장치용 미들웨어를 추가로 구현했으며, `agent.py`의
`middleware=[...]` 목록에 아래 순서대로 연결되어 있습니다.

### 1. Emergency Screening Middleware (`@before_agent`)

```python
@before_agent
def emergency_screening_middleware(state, runtime):
    # 사용자의 최신 메시지에서 응급 키워드를 선제적으로 검사
    # 감지되면 LLM이 도구를 호출하기 전에
    # "응급 안내를 최우선으로 답변하라"는 SystemMessage를 주입
```

- `search_symptom_info` 도구 내부의 응급 감지 로직과는 별개로, **에이전트 진입 시점**에
  한 번 더 선제 검사하여 이중 안전장치를 둡니다.

### 2. Skill Injection Middleware (`@before_agent`)

```python
@before_agent
def skill_injection_middleware(state, runtime):
    # 응급이 아닌 일반 증상 언급이 감지되면
    # skills/symptom-navigation/skill.md 절차를 SystemMessage로 자동 주입
    # (동일 대화 내 중복 주입 방지, 응급 감지 시에는 건너뜀)
```

- `emergency_screening_middleware` 바로 다음 순서로 실행되어, 응급이 아닌 경우에만
  6단계 절차를 주입합니다.

### 3. Disclaimer Guard Middleware (`@wrap_tool_call`)

```python
@wrap_tool_call
async def disclaimer_guard_middleware(request, handler):
    # search_symptom_info / lookup_drug_info / get_health_info 결과에
    # "참고용" 안내 문구가 누락되어 있으면 자동으로 보강
```

### 4. Tool Call Logging Middleware (`@wrap_tool_call`)

```python
@wrap_tool_call
async def tool_call_logging_middleware(request, handler):
    # 모든 도구 호출(도구명, 입력값, 시각)을 logs/tool_calls.log 에 기록
    # 의료 정보 오남용 점검 및 감사(audit) 목적
```

---

## 테스트 질문 예시

```
- "두통과 어지러움이 있는데 원인이 뭘까?"          # symptom-navigation skill 절차 트리거
- "가슴이 아프고 숨쉬기 힘들어"                     # 응급 감지 테스트 (skill 절차 건너뜀)
- "천안시 서북구 내과 병원 알려줘"
- "타이레놀 부작용 알려줘"
- "고혈압 예방법 알려줘"
```

---

## LangGraph Studio 실행 화면 / 결과물 예시
<img width="3280" height="2460" alt="image" src="https://github.com/user-attachments/assets/c7e50de0-872b-4df4-8923-800f01ce1f12" />
<img width="3280" height="2460" alt="image" src="https://github.com/user-attachments/assets/529a50e8-267a-45bd-b039-c655d0d4cca3" />

```

---

## 참고 자료

- [LangChain Tools 문서](https://python.langchain.com/docs/modules/agents/tools/)
- [LangChain Custom Tools](https://python.langchain.com/docs/modules/agents/tools/custom_tools/)
- [LangGraph Middleware](https://langchain-ai.github.io/langgraph/)
