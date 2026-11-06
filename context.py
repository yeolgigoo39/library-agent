"""에이전트 실행에 필요한 사서·이용자 정보."""

from dataclasses import dataclass


@dataclass
class Context:
    """이번 실행의 사서·이용자. 도구 인자로 열지 않는다."""

    librarian_name: str
    librarian_role: str
    user_name: str
    user_id: str
    app_name: str
