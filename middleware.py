"""미들웨어와 장기 기억.

05_full_agent.py와 같은 순서다.
  1) library_policy_guardrail  업무 무관·타인 대출 차단
  2) route_library_info         시설·규정 문의를 도구로 직접 연결
  3) enrich_context             사서 이름·역할과 장기 기억 주입
  4) answer_audit_guardrail     타인 대출 정보 포함 여부 검수
"""

import uuid
from typing import Callable, TypedDict

from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    after_agent,
    before_agent,
    wrap_model_call,
)
from langchain.messages import AIMessage
from langchain.tools import ToolRuntime, tool
from langgraph.store.memory import InMemoryStore

from catalog import book_by_title, compact, loans
from context import Context
from tools import get_library_info, library_tool_requests


class UserInfo(TypedDict):
    """장기 기억에 저장하는 이용자 정보."""

    personal_information: str
    preference: str


# 대화창이 바뀌어도 남는 저장소
store = InMemoryStore()
store.put(
    ("202258036", "library_assistant"),
    "seed",
    {
        "facts": [
            "이용자 이준석(202258036)은 판타지와 추리 소설을 선호함",
            "대출 가능 여부, 예약 순번, 열람실 자리는 도구로 확인한 값만 안내함",
            "목록에 없는 책은 있다고 말하지 않음",
        ]
    },
)

ctx = Context(
    librarian_name="한소영",
    librarian_role="한신대학교 도서관 사서 보조",
    user_name="이준석",
    user_id="202258036",
    app_name="library_assistant",
)

forbidden_topics = {
    "off_topic": ["롤 해줘", "게임 추천", "주식", "날씨 알려", "요리법", "유튜브 추천", "농담만"],
    "privacy": ["다른 사람 대출", "다른 이용자 대출", "타인 대출", "전체 대출", "회원 대출 목록", "남의 대출"],
    "abusive": ["바보", "멍청이", "꺼져"],
}
BLOCK_MESSAGES = {
    "off_topic": "도서관 자료 검색, 대출·예약, 이용 안내만 도와드릴 수 있습니다.",
    "privacy": "다른 이용자의 대출 정보는 안내할 수 없습니다. 본인 대출만 조회할 수 있습니다.",
    "abusive": "불편을 드려 죄송합니다. 도서관 이용 문의를 다시 말씀해 주세요.",
}


@before_agent(can_jump_to=["end"])
def library_policy_guardrail(state, runtime):
    """모델 호출 전에 업무 무관 질문과 타인 대출 조회를 끊는다."""
    if not state["messages"]:
        return None
    last = state["messages"][-1]
    if getattr(last, "type", None) != "human":
        return None
    text = last.content or ""
    my_id = getattr(runtime.context, "user_id", "")
    other_names = [row["user_name"] for row in loans if row["user_id"] != my_id and row["user_name"]]
    q = compact(text)
    if any(name in text for name in other_names) and any(word in text for word in ("대출", "반납", "연체", "기록", "예약", "빌린", "빌렸")):
        return {"messages": [AIMessage(content=BLOCK_MESSAGES["privacy"])], "jump_to": "end"}
    if any(word in q for word in ("다른이용자", "다른사람", "타인", "모든회원", "전체회원")) and any(word in q for word in ("빌린", "대출", "예약", "반납")):
        return {"messages": [AIMessage(content=BLOCK_MESSAGES["privacy"])], "jump_to": "end"}
    for category, keywords in forbidden_topics.items():
        if any(keyword in text for keyword in keywords):
            # 주식이라는 주제의 도서 검색은 투자 조언과 구분한다.
            book_request = any(word in q for word in ("도서", "책", "작가", "장르")) and any(
                req["name"] in ("search_books", "recommend_books") for req in library_tool_requests(text)
            )
            if category == "off_topic" and book_request and not any(word in q for word in ("종목", "매수", "매도", "수익률", "날씨알려", "농담만")):
                continue
            return {"messages": [AIMessage(content=BLOCK_MESSAGES[category])], "jump_to": "end"}
    return None


@before_agent(can_jump_to=["end", "tools"])
def route_library_info(state, runtime):
    """시설 단독 문의는 바로 답하고, 명시된 복합 작업은 기존 도구 노드로 연결한다."""
    if not state["messages"]:
        return None
    last = state["messages"][-1]
    if getattr(last, "type", None) != "human":
        return None
    text = last.content if isinstance(last.content, str) else ""
    requests = library_tool_requests(text)
    q = compact(text)
    if not requests:
        if any(word in q for word in ("예약하지마", "예약하지말", "예약하지않", "예약은하지마", "예약은하지않")):
            return {"messages": [AIMessage(content="예약을 신청하지 않았습니다.")], "jump_to": "end"}
        return None
    # 본인이 빌린 책의 작가 검색은 선호 정보가 아니라 본인 대출 자료를 사용한다.
    if "내가빌린" in q and "작가" in q:
        authors = list(dict.fromkeys(
            book["author"] for row in loans
            if row["user_id"] == runtime.context.user_id and "대출" in row["status"]
            if (book := book_by_title(row["title"])) is not None
        ))
        for request in requests:
            if request["name"] == "search_books":
                request["args"]["query"] = " ".join(authors)
    if [request["name"] for request in requests] == ["get_library_info"]:
        answer = get_library_info.invoke({"query": text})
        return {"messages": [AIMessage(content=answer)], "jump_to": "end"}
    calls = [dict(request, id=str(uuid.uuid4()), type="tool_call") for request in requests]
    return {"messages": [AIMessage(content="", tool_calls=calls)], "jump_to": "tools"}


@wrap_model_call
def enrich_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """사서 이름·역할과 장기 기억을 이번 모델 입력에만 넣는다."""
    context = request.runtime.context
    librarian_name = getattr(context, "librarian_name", "") or ""
    librarian_role = getattr(context, "librarian_role", "") or ""
    user_name = getattr(context, "user_name", "") or ""
    memories = request.runtime.store.search((context.user_id, context.app_name)) if request.runtime.store else []
    facts = []
    for item in memories:
        facts.extend(item.value.get("facts", []))
        for field in ("personal_information", "preference"):
            if item.value.get(field):
                facts.append(item.value[field])
    memory_text = "\n- ".join(facts) if facts else "기록된 정보 없음"
    prompt = (
        f"{request.system_prompt or ''}\n"
        f"당신은 {librarian_role}입니다. 사서 이름은 {librarian_name}입니다.\n"
        f"지금 응대 중인 이용자는 {user_name}입니다.\n"
        f"[장기 메모리]\n- {memory_text}\n"
        "목록에 없는 책은 있다고 말하지 마세요. 다른 이용자의 대출 정보는 말하지 마세요."
    ).strip()
    return handler(request.override(system_prompt=prompt))


@after_agent
def answer_audit_guardrail(state, runtime):
    """최종 답변에 다른 이용자의 대출 정보가 포함되면 안전한 문장으로 바꾼다."""
    if not state["messages"]:
        return None
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not (last_message.content or "").strip():
        return None
    # 확인된 도서관 결과를 모두 보존한다. 여러 검색·시설·추천의 일부가 빠지지 않게 한다.
    turn_tools = []
    for message in reversed(state["messages"][:-1]):
        if getattr(message, "type", None) == "human":
            break
        if getattr(message, "type", None) == "tool":
            turn_tools.append(message)
    factual_tools = {"search_books", "get_book_status", "reserve_book", "check_loan_tool", "get_library_info", "recommend_books"}
    if turn_tools and all(
        getattr(message, "name", "") in factual_tools and getattr(message, "status", "success") == "success"
        for message in turn_tools
    ):
        results = list(dict.fromkeys(message.content for message in reversed(turn_tools)))
        last_message.content = "\n\n".join(results)
    answer = last_message.content
    user_id = runtime.context.user_id
    for row in loans:
        if row["user_id"] == user_id:
            continue
        if row["user_name"] and row["user_name"] in answer and any(
            word in answer for word in ("대출", "반납", "연체", "예약")
        ):
            last_message.content = (
                "다른 이용자의 대출 정보는 안내할 수 없습니다. "
                "본인 대출이나 도서관 자료를 다시 물어봐 주세요."
            )
            break
    return None


@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """현재 이용자의 저장된 이름·선호를 조회한다."""
    memories = runtime.store.search((runtime.context.user_id, runtime.context.app_name))
    if not memories:
        return "기록된 정보 없음"
    lines = []
    for item in memories:
        data = item.value
        if "personal_information" in data:
            lines.append(f"- 개인정보: {data['personal_information']}")
        if "preference" in data:
            lines.append(f"- 선호도: {data['preference']}")
        if "facts" in data:
            lines.extend(f"- {fact}" for fact in data["facts"])
    return "\n".join(lines) if lines else "형식 불일치"


@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """이용자 이름·독서 선호를 장기 메모리에 저장한다."""
    key = str(uuid.uuid4())
    runtime.store.put((runtime.context.user_id, runtime.context.app_name), key, user_info)
    return f"정보가 안전하게 저장되었습니다. (ID: {key})"


MEMORY_TOOLS = [get_user_info, save_user_info]
MIDDLEWARE = [
    library_policy_guardrail,
    route_library_info,
    enrich_context,
    answer_audit_guardrail,
]
