"""
의료 정보 제공 및 안내 에이전트 - 커스텀 도구 모음

이 파일은 두 부분으로 구성됩니다.
1. Day7 실습(day7_team_project_template.ipynb)에서 AI 프롬프트로 생성하고
   정상/엣지/에러 케이스 테스트를 마친 4개의 기본 도구
   (search_symptom_info, find_hospital, lookup_drug_info, get_health_info)
2. LangGraph Studio 연동 및 Claude Code 활성화 단계에서 symptom-navigation
   skill(skills/symptom-navigation/skill.md)을 지원하기 위해 추가한 4개 도구
   (recommend_department, find_hospital 보강, check_hospital_hours,
   generate_visit_checklist)와 skill.md 로더(load_skill)
"""

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# ---------------------------------------------------------------------------
# 공통 상수
# ---------------------------------------------------------------------------
TRUSTED_MEDICAL_DOMAINS = [
    "amc.seoul.kr",
    "snuh.org",
    "health.kdca.go.kr",
    "nedrug.mfds.go.kr",
    "mayoclinic.org",
    "webmd.com",
]

DRUG_INFO_DOMAINS = [
    "nedrug.mfds.go.kr",
    "druginfo.co.kr",
    "kimsonline.co.kr",
]

# 응급 가능성이 있는 증상 키워드 (검색 없이 즉시 119/응급실 안내)
EMERGENCY_KEYWORDS = [
    "가슴 통증", "호흡 곤란", "의식 없음", "심한 출혈",
    "마비", "발작", "고열 40도", "심한 복통", "실신",
]

# 일반 증상 감지용 키워드 (응급 키워드보다 넓은 범위 - "증상을 말했다"는
# 신호만 잡아내어 skill_injection_middleware가 symptom-navigation skill을
# 주입할지 판단하는 데 사용)
SYMPTOM_KEYWORDS = [
    "아파요", "아프다", "아픈", "통증", "쑤셔", "쑤시",
    "열이", "발열", "미열",
    "기침", "가래", "콧물", "코막힘", "재채기",
    "두통", "머리가", "어지러", "어지럽",
    "복통", "배가", "속이", "메스꺼", "구토", "설사", "변비",
    "발진", "가려워", "가렵", "붓기", "부었", "부어",
    "몸살", "근육통", "허리가", "무릎이", "손목이", "발목이",
    "목이 아파", "목이 부", "귀가 아파",
    "숨이", "호흡이",
]

DISCLAIMER = (
    "\n\n⚠️ 본 정보는 참고용이며 진단·처방을 대체하지 않습니다. "
    "정확한 진단과 치료는 반드시 의료진과 상담하세요."
)

EMERGENCY_NOTICE = (
    "\n\n🚨 응급 가능성이 있는 증상입니다. "
    "즉시 119에 신고하거나 가까운 응급실을 방문하세요."
)


def _check_emergency(text: str) -> bool:
    return any(keyword in text for keyword in EMERGENCY_KEYWORDS)


# ---------------------------------------------------------------------------
# skill.md 로더
# - LLM의 tool-call 판단 없이 middleware.py의 skill_injection_middleware가
#   직접 호출해서 사용합니다.
# ---------------------------------------------------------------------------
SKILLS_DIR = Path(__file__).parent / "skills"


def load_skill(skill_name: str = "symptom-navigation") -> str:
    """skills/<skill_name>/skill.md 파일을 읽어 문자열로 반환합니다.

    파일이 없으면 빈 문자열을 반환합니다 (호출부에서 실패 처리).
    """
    skill_path = SKILLS_DIR / skill_name / "skill.md"
    try:
        return skill_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------------------
# 1) Day7 실습에서 생성한 기본 4개 도구
# ---------------------------------------------------------------------------

@tool(parse_docstring=True)
def search_symptom_info(symptom: str) -> str:
    """증상을 기반으로 가능한 원인, 대처법, 병원 방문 권고 시점을 신뢰 가능한
    의학 정보 출처에서 검색합니다. 진단을 내리지 않습니다.

    Args:
        symptom: 검색할 증상 (예: "두통과 어지러움")

    Returns:
        증상 관련 정보 요약 및 주의사항, 또는 응급 안내 메시지
    """
    try:
        if not symptom:
            return "증상을 입력해 주세요."

        if _check_emergency(symptom):
            return f"입력하신 증상({symptom})은 응급 상황일 수 있습니다.{EMERGENCY_NOTICE}"

        search = TavilySearch(max_results=5, include_domains=TRUSTED_MEDICAL_DOMAINS)
        results = search.invoke({"query": f"{symptom} 원인 증상 대처법"})
        return f"[증상 검색 결과: {symptom}]\n{results}{DISCLAIMER}"
    except Exception as e:
        return f"실패: {str(e)}"


@tool(parse_docstring=True)
def find_hospital(location: str, department: Optional[str] = None) -> str:
    """지역과 진료과를 기반으로 병원 정보를 검색합니다.

    Args:
        location: 검색할 지역 (예: "천안시 서북구")
        department: 진료과 (예: "내과"). 없으면 종합 검색.

    Returns:
        검색된 병원 목록 및 기본 정보
    """
    try:
        if not location:
            return "지역 정보를 입력해 주세요. (예: '천안시 서북구')"

        dept_query = f"{department} " if department else ""
        query = f"{location} {dept_query}병원 진료시간 전화번호"

        search = TavilySearch(max_results=5)
        results = search.invoke({"query": query})
        return (
            f"[병원 검색 결과: {location} / {department or '전체'}]\n"
            f"{results}\n\n※ 방문 전 반드시 병원에 전화로 진료 가능 여부를 확인하세요."
        )
    except Exception as e:
        return f"실패: {str(e)}"


@tool(parse_docstring=True)
def lookup_drug_info(drug_name: str) -> str:
    """의약품명을 기반으로 효능, 용법·용량, 주의사항, 부작용 정보를
    공식 의약품 데이터베이스에서 조회합니다.

    Args:
        drug_name: 조회할 약품명 (예: "타이레놀")

    Returns:
        의약품 정보 요약, 또는 정보 없음 안내 메시지
    """
    try:
        if not drug_name:
            return "약품명을 입력해 주세요."

        search = TavilySearch(max_results=3, include_domains=DRUG_INFO_DOMAINS)
        results = search.invoke({"query": f"{drug_name} 효능 용법 주의사항 부작용"})

        if not results:
            return f"'{drug_name}'에 대한 공식 정보를 찾지 못했습니다. 약사 또는 의사와 상담하세요."

        return (
            f"[의약품 정보: {drug_name}]\n{results}\n\n"
            f"⚠️ 다른 약물과의 상호작용은 반드시 약사와 상담하세요."
        )
    except Exception as e:
        return f"실패: {str(e)}"


@tool(parse_docstring=True)
def get_health_info(topic: str) -> str:
    """건강 관리, 예방, 생활 습관 등 일반 건강 정보를 검색하여 제공합니다.

    Args:
        topic: 검색할 건강 주제 (예: "고혈압 예방법")

    Returns:
        건강 정보 요약
    """
    try:
        if not topic:
            return "건강 주제를 입력해 주세요. (예: '고혈압 예방법')"

        search = TavilySearch(max_results=5, include_domains=TRUSTED_MEDICAL_DOMAINS)
        results = search.invoke({"query": topic})
        return f"[건강 정보: {topic}]\n{results}{DISCLAIMER}"
    except Exception as e:
        return f"실패: {str(e)}"


# ---------------------------------------------------------------------------
# 2) symptom-navigation skill(skills/symptom-navigation/skill.md)이 참조하는
#    보조 도구 - recommend_department, check_hospital_hours,
#    generate_visit_checklist는 skill의 절차 단계를 지원하기 위해 추가되었습니다.
#    recommend_department, check_hospital_hours는 실제 데이터 소스(공공 API 등)
#    연동이 필요한 부분에 TODO로 표시했습니다.
#    generate_visit_checklist는 외부 연동 없이 바로 동작하는 결정론적 도구입니다.
# ---------------------------------------------------------------------------

# 증상 키워드 -> 진료과 매핑 (1차 버전: 필요에 따라 세분화/모델 기반으로 교체 가능)
_DEPARTMENT_RULES: list[tuple[list[str], str]] = [
    (["기침", "가래", "콧물", "코막힘", "재채기", "목이 아파", "목이 부"], "이비인후과"),
    (["복통", "배가", "속이", "메스꺼", "구토", "설사", "변비"], "소화기내과"),
    (["두통", "머리가", "어지러", "어지럽"], "신경과"),
    (["발진", "가려워", "가렵", "붓기", "부었", "부어"], "피부과"),
    (["몸살", "근육통", "허리가", "무릎이", "손목이", "발목이"], "정형외과"),
    (["열이", "발열", "미열"], "내과"),
]


@tool(parse_docstring=True)
def recommend_department(symptom: str) -> str:
    """사용자가 말한 증상 텍스트를 바탕으로 적절한 진료과를 1~2개 추천합니다.

    Args:
        symptom: 사용자가 언급한 증상 설명 텍스트

    Returns:
        추천 진료과 목록 (쉼표로 구분된 문자열)
    """
    matched = []
    for keywords, department in _DEPARTMENT_RULES:
        if any(keyword in symptom for keyword in keywords) and department not in matched:
            matched.append(department)

    if not matched:
        matched.append("내과")  # 매칭되는 규칙이 없으면 내과를 기본값으로 안내

    return ", ".join(matched[:2])


@tool(parse_docstring=True)
def check_hospital_hours(hospital_name: str) -> str:
    """지정된 병원의 오늘 진료 여부와 진료 시간을 확인합니다.

    TODO: find_hospital과 동일한 외부 API에서 영업시간 필드를 함께 받아오거나,
    별도 API 호출로 오늘 요일 기준 진료 시간을 조회하도록 구현해야 합니다.

    Args:
        hospital_name: 진료 여부를 확인할 병원 이름

    Returns:
        해당 병원의 오늘 진료 여부 안내 (임시 응답)
    """
    # TODO: 외부 API 연동 (건강보험심사평가원 병원정보서비스 등)
    return (
        f"[임시 응답] '{hospital_name}'의 오늘 진료 여부 확인 기능은 "
        "아직 외부 API와 연동되지 않았습니다."
    )


@tool(parse_docstring=True)
def generate_visit_checklist(symptom: str, department: str) -> str:
    """병원 방문 전 준비물 및 확인사항 체크리스트를 생성합니다.

    Args:
        symptom: 사용자가 언급한 증상
        department: 방문 예정 진료과

    Returns:
        방문 전 체크리스트 텍스트
    """
    checklist = [
        "신분증",
        "건강보험증",
        "복용 중인 약물 목록 (있는 경우)",
        f"증상 메모: '{symptom}' - 발생 시점, 양상, 변화 추이를 적어가면 진료에 도움이 됩니다.",
        f"{department} 사전 예약 필요 여부 확인",
    ]
    return "방문 전 체크리스트:\n" + "\n".join(f"- {item}" for item in checklist)


# ---------------------------------------------------------------------------
# 에이전트에 등록할 전체 도구 목록
# ---------------------------------------------------------------------------
CUSTOM_TOOLS = [
    search_symptom_info,
    find_hospital,
    lookup_drug_info,
    get_health_info,
    recommend_department,
    check_hospital_hours,
    generate_visit_checklist,
]
