"""에이전트 도구.

검색, 추천, 대출 상태, 예약, 본인 대출, 시설·규정, 회신을 도구로 제공한다.
확인하지 않은 책 이름은 NOT_FOUND 문장만 돌려준다.
"""

import re

from langchain.tools import ToolRuntime, tool

from catalog import (
    NOT_FOUND,
    book_block,
    compact,
    find_books,
    inventory,
    library_rules,
    loans,
    reading_rooms,
    sync_loan_status,
    title_matches,
)
from context import Context
from recommend import rank_recommendations


@tool
def search_books(query: str) -> str:
    """소장 도서를 검색한다. 제목, 작가, 출판사, 장르, 키워드, 이용자 문장을 그대로 받는다."""
    if not (query or "").strip():
        return "검색어가 없습니다. 제목, 작가, 장르를 알려 주세요."
    found = find_books(query)
    if not found:
        return NOT_FOUND
    return "검색 결과:\n\n" + "\n\n".join(book_block(book) for book in found)


@tool
def recommend_books(
    query: str,
    runtime: ToolRuntime[Context],
    exclude_same_author: bool = True,
    use_trends: bool = True,
    use_vibe: bool = True,
) -> str:
    """다른 작가의 책을 추천한다. 독자 동향을 먼저 보고 감성 유사도로 순서를 정한다."""
    read_titles = [row["title"] for row in loans if row["user_id"] == runtime.context.user_id]
    titled = [book for book in find_books(query) if compact(book["title"]) in compact(query)]
    if titled:
        anchors = [book["title"] for book in titled]
    elif read_titles and any(word in query for word in ("추천", "비슷", "다른 책", "다음 책")):
        anchors = read_titles
    else:
        found = find_books(query)
        anchors = [found[0]["title"]] if found else read_titles
    return rank_recommendations(anchors, read_titles, exclude_same_author, use_trends, use_vibe)


@tool
def get_book_status(query: str) -> str:
    """대출 가능 여부와 예약 대기 인원을 조회한다."""
    found = find_books(query)
    if not found:
        return NOT_FOUND
    lines = []
    for book in found:
        item = inventory[book["title"]]
        lines.append(book_block(book))
        lines.append(f"예약 대기: {len(item['waitlist'])}명\n")
    return "\n".join(lines)


@tool
def reserve_book(query: str, runtime: ToolRuntime[Context]) -> str:
    """현재 이용자 이름으로 대출 중인 도서의 예약 대기를 신청한다."""
    # 모델이 방법 문의나 부정문을 신청으로 오해하더라도 실제 예약은 만들지 않는다.
    messages = getattr(runtime, "state", {}).get("messages", [])
    original = next((message.content for message in reversed(messages) if getattr(message, "type", None) == "human"), None)
    if original is not None and not any(request["name"] == "reserve_book" for request in library_tool_requests(original)):
        return "예약은 신청 의사를 확인한 뒤 처리합니다. 예약하려면 도서 제목과 함께 예약해 달라고 말씀해 주세요."
    found = title_matches(query)
    if not found:
        if find_books(query):
            return "예약할 도서의 정확한 제목을 하나만 지정해 주세요."
        return NOT_FOUND
    if len(found) > 1:
        titles = ", ".join(book["title"] for book in found)
        return f"여러 권이 검색되었습니다. 제목을 하나만 지정해 주세요: {titles}"
    if original is not None:
        requested = title_matches(original)
        if len(requested) != 1 or requested[0]["title"] != found[0]["title"]:
            return "예약 대상이 질문의 도서와 일치하지 않습니다. 정확한 제목을 하나만 지정해 주세요."
    book = found[0]
    item = inventory[book["title"]]
    user_id = runtime.context.user_id
    user_name = runtime.context.user_name
    already = user_id in item["waitlist"] or any(
        row["user_id"] == user_id and row["title"] == book["title"] and "예약" in row["status"]
        for row in loans
    )
    if already:
        return f"{book['title']}은 이미 예약되어 있습니다."
    if item["available"] > 0:
        return f"{book['title']}은 대출 가능합니다. 규정에 따라 예약 없이 대출해 주세요."
    item["waitlist"].append(user_id)
    order = len(item["waitlist"])
    loans.append(
        {
            "user_id": user_id,
            "user_name": user_name,
            "title": book["title"],
            "due": "대기",
            "status": f"예약대기 {order}번",
        }
    )
    return f"{book['title']}은 지금 대출할 수 없습니다. 예약 대기 {order}번으로 등록했습니다."


@tool
def check_loan_tool(runtime: ToolRuntime[Context]) -> str:
    """현재 이용자의 대출·예약만 조회한다. 이용자 id는 모델이 넣지 못한다."""
    mine = [row for row in loans if row["user_id"] == runtime.context.user_id]
    if not mine:
        return "현재 이용자의 대출·예약 기록이 없습니다."
    lines = [f"- {row['title']} / {row['status']} / {row['due']}" for row in mine]
    return "내 대출·예약:\n" + "\n".join(lines)


@tool
def get_library_info(query: str) -> str:
    """운영시간, 열람실 잔여석, 대출·반납 규정을 조회한다."""
    q = compact(query)
    want_rules, want_hours, want_seats = library_info_topics(query)
    if not want_hours and not want_seats and not want_rules:
        return "운영 시간, 열람실 자리, 대출·반납 규정만 안내할 수 있습니다."
    lines = []
    if want_rules or want_hours:
        lines.append(library_rules)
    if want_seats:
        picked = [room for room in reading_rooms if compact(room["name"]) in q]
        if not picked and "열람실" in q and not any(ch.isdigit() for ch in q):
            picked = list(reading_rooms)
        if not picked:
            lines.append("해당 열람실을 찾을 수 없습니다. 1열람실, 2열람실, 3열람실만 있습니다.")
        for room in picked:
            lines.append(f"{room['name']}: 전체 {room['total']}석, 남은 자리 {room['available']}석.")
    return "\n".join(lines)


@tool
def reply_to_user_tool(to: str, subject: str, body: str) -> str:
    """이용자 안내 회신을 기록한다. 실제 발송은 하지 않는다."""
    return f"회신 기록 완료. to={to}, subject={subject}"


def library_info_topics(query: str) -> tuple[bool, bool, bool]:
    """문의가 규정, 운영시간, 열람실 중 무엇에 해당하는지 반환한다."""
    # 제목 속 '닫힌', '폐관' 등을 시설 질문으로 취급하지 않는다.
    q = compact(query)
    for book in title_matches(query):
        q = q.replace(compact(book["title"]), "")
    want_rules = any(word in q for word in ("규정", "연체", "대출기간", "연장", "몇권", "늦게돌려", "늦게반납", "예약방법", "예약할수", "예약가능", "예약해도", "예약하려면"))
    personal_return = any(word in q for word in ("반납일", "반납예정", "언제반납", "언제까지돌려", "언제돌려"))
    want_rules = want_rules or ("반납" in q and not personal_return)
    want_hours = any(word in q for word in ("몇시", "열어", "운영", "휴관", "문닫", "닫아", "개관", "폐관"))
    want_seats = any(word in q for word in ("열람실", "자리", "좌석"))
    return want_rules, want_hours, want_seats


def library_tool_requests(query: str) -> list[dict]:
    """명시된 도서관 작업을 빠짐없이 연결한다. 불명확한 후속 질문은 모델에 맡긴다."""
    q = compact(query)
    requests = []
    def add(name):
        requests.append({"name": name, "args": {} if name == "check_loan_tool" else {"query": query}})

    own = any(word in q for word in ("내대출", "내예약", "내가빌린", "내가읽은", "반납일", "반납예정", "언제반납", "언제까지돌려", "언제돌려"))
    recommend = "추천" in q or "비슷한" in q
    search = any(word in q for word in ("검색", "찾아", "작품"))
    status = any(word in q for word in ("빌릴수", "대출가능", "대출상태", "재고")) and "몇권" not in q
    # 신청 의사를 명시한 경우만 연결한다. 조건·부정·방법 문의는 신청이 아니다.
    reserve = bool(re.search(r"예약(?:을)?(?:좀)?(?:해주세요|해줘|해주실래|부탁|신청해|신청할게|할게|할래|하고싶|하자|해$)", q))
    reserve = reserve and not any(word in q for word in ("하지마", "하지않", "말고", "취소", "하면", "라면", "아니면", "방법"))
    if own:
        add("check_loan_tool")
    if any(library_info_topics(query)):
        add("get_library_info")
    if reserve:
        add("reserve_book")
    if recommend:
        add("recommend_books")
    if status and not reserve:
        add("get_book_status")
    if search and (not recommend or any(word in q for word in ("검색", "찾아"))):
        add("search_books")
    return requests


# create_agent에 넘길 도구. 장기 기억 도구는 middleware.py에 있다.
LIBRARY_TOOLS = [
    search_books,
    recommend_books,
    get_book_status,
    reserve_book,
    check_loan_tool,
    get_library_info,
    reply_to_user_tool,
]
