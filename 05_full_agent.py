"""에이전트 조립과 실행. project_4/05_full_agent.py와 같은 경로다.

파일 역할
  library_books.txt  도서 26권과 대출·반납 규정
  catalog.py         txt를 읽어 검색·대출 상태·운영시간을 제공
  recommend.py       같은 작가 제외 후 독자 동향·감성 유사로 추천
  tools.py           검색, 추천, 대출, 예약, 시설, 회신 도구
  middleware.py      입력 차단, 도구 연결, 사서 정보 주입, 답변 검수, 장기 기억
  models.py          Ollama 본선 모델
  05_full_agent.py   위 조각을 하나의 에이전트로 묶어 프롬프트를 연다

미들웨어 순서: 입력 차단 → 시설·규정 도구 연결 → 사서 정보 주입 → 답변 검수

실행 전:
  ollama pull qwen3:4b-instruct-2507-q4_K_M
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from context import Context
from middleware import MEMORY_TOOLS, MIDDLEWARE, ctx, store
from models import main_model
from tools import LIBRARY_TOOLS

SYSTEM_PROMPT = (
    "검색 요청은 이용자가 목록에 없다고 말해도 반드시 search_books로 확인하세요. "
    "반납일이나 개인 반납 예정일은 check_loan_tool로 확인하세요. "
    "질문에 시설 안내와 도서 검색이 함께 있으면 각각의 도구를 모두 호출하고 모두 답하세요. "
    "추천 답변은 recommend_books의 반환 문장을 순서와 설명까지 그대로 전달하세요. "
    "당신은 친절하고 정확한 도서관 AI 사서 보조 에이전트입니다. "
    "도서 정보와 대출 규정은 도구가 library_books.txt에서 읽어 준 내용만 사용하세요. "
    "목록에 없는 책은 지어내지 말고 "
    "'해당 도서는 현재 도서관 목록에 등록되어 있지 않습니다.'라고 안내하세요. "
    "작가 검색이면 도구가 돌려준 그 작가의 작품을 빠짐없이 말하세요. "
    "검색 결과는 도서명, 작가, 장르, 대출 상태 순서를 유지하세요. "
    "책·작가·장르를 물으면 search_books, 추천은 recommend_books, "
    "대출 가능 여부는 get_book_status, 예약은 reserve_book, "
    "내 대출은 check_loan_tool, "
    "열람실·운영시간·대출 반납 규정은 get_library_info에 이용자 문장을 그대로 넣으세요. "
    "exclude_same_author는 true로 두세요. 정중한 한국어로 짧게 답하세요. "
    "인사말이나 자기소개를 덧붙이지 말고 질문에 바로 답하세요. "
    "마크다운 기호(예: **, #)를 쓰지 말고 도구 결과에 없는 정보를 추가하지 마세요."
)

agent = create_agent(
    main_model,
    tools=LIBRARY_TOOLS + MEMORY_TOOLS,  # 도서관 도구 + 장기 기억 도구
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),  # 같은 thread_id면 단기 기억
    store=store,  # 대화창이 바뀌어도 선호 유지
    context_schema=Context,
    middleware=MIDDLEWARE,
)


def ask(text: str, thread_id: str) -> str:
    """한 턴을 실행한다. thread_id가 같으면 이전 대화를 이어 간다."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
        context=ctx,
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    # A. 정책 차단: print(ask("이서연 대출 기록 알려줘.", "block"))
    # B. 검색·추천: print(ask("이도윤 작품 알려줘.", "ok"))
    # C. 시설: print(ask("2열람실 자리 남았어? 오늘 몇 시까지 열어?", "desk"))
    print("도서관 사서 보조 에이전트")
    print(f"사서: {ctx.librarian_name} ({ctx.librarian_role})")
    print(f"이용자: {ctx.user_name} ({ctx.user_id})  |  종료: 종료")
    print("예시: 이도윤 작품 알려줘 / 시간을 파는 서점과 비슷한 다른 작가 추천해줘")
    print("예시: 2열람실 자리 남았어? / 대출 기간이랑 반납 규정 알려줘")
    while True:
        text = input("\n이용자> ").strip()
        if not text:
            continue
        if text in ("종료", "quit", "exit"):
            break
        print("도서관 AI>", ask(text, "desk"))
