"""도서 목록과 도서관 상태.

library_books.txt를 읽어 검색하고, 대출 상태·운영시간·열람실·본인 대출 기록을 보관한다.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

# 03_rag.py와 같이, 이 파일과 같은 폴더의 텍스트를 읽는다.
CATALOG_PATH = Path(__file__).with_name("library_books.txt")
NOT_FOUND = "해당 도서는 현재 도서관 목록에 등록되어 있지 않습니다."

# 열람실 잔여석. 조회 도구가 이 숫자를 그대로 말한다.
reading_rooms = [
    {"name": "1열람실", "total": 120, "available": 14},
    {"name": "2열람실", "total": 80, "available": 6},
    {"name": "3열람실", "total": 40, "available": 0},
]

# 본인 대출만 도구로 나간다. 이서연 기록은 타인 조회를 막는 가드레일 확인용이다.
loans: List[Dict[str, str]] = [
    {
        "user_id": "202258036",
        "user_name": "이준석",
        "title": "시간을 파는 서점",
        "due": "2026-11-20",
        "status": "대출중",
    },
    {
        "user_id": "202200001",
        "user_name": "이서연",
        "title": "달빛 도서관의 비밀",
        "due": "2026-11-12",
        "status": "대출중",
    },
]


def _split_csv(text: str) -> List[str]:
    return [part.strip() for part in (text or "").split(",") if part.strip()]


def load_catalog(path: Path = CATALOG_PATH):
    """txt의 [도서]와 [규정]을 목록, 재고, 규정 문장으로 나눈다."""
    raw = path.read_text(encoding="utf-8")
    book_part, _, rule_part = raw.partition("[규정]")
    loaded = []
    stock: Dict[str, Dict] = {}
    for block in book_part.split("[도서]"):
        fields = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
        title = fields.get("도서명", "")
        if not title:
            continue
        status = "대출 중" if "중" in fields.get("대출 상태", "") else "대출 가능"
        loaded.append(
            {
                "title": title,
                "author": fields.get("작가", ""),
                "publisher": fields.get("출판사", ""),
                "genre": fields.get("장르", ""),
                "keywords": _split_csv(fields.get("키워드", "")),
                "tags": _split_csv(fields.get("태그", "")),
                "synopsis": fields.get("줄거리", ""),
                "loan_status": status,
            }
        )
        stock[title] = {"total": 1, "available": 0 if status == "대출 중" else 1, "waitlist": []}
    if not loaded:
        raise RuntimeError(f"도서 목록이 비어 있습니다: {path}")
    return loaded, stock, rule_part.strip()


books, inventory, library_rules = load_catalog()


def compact(text: str) -> str:
    """공백을 뺀 비교용 문자열."""
    return "".join((text or "").lower().split())


def book_by_title(title: str) -> Optional[dict]:
    for book in books:
        if book["title"] == title:
            return book
    return None


def sync_loan_status(title: str) -> None:
    """예약으로 남은 권수가 바뀌면 목록의 대출 상태 문구도 맞춘다."""
    book = book_by_title(title)
    if book is None:
        return
    book["loan_status"] = "대출 가능" if inventory[title]["available"] > 0 else "대출 중"


def book_block(book: dict) -> str:
    """검색 결과 한 권. 도서명, 작가, 장르, 대출 상태 순서."""
    sync_loan_status(book["title"])
    return (
        f"도서명: {book['title']}\n"
        f"작가: {book['author']}\n"
        f"장르: {book.get('genre', '')}\n"
        f"대출 상태: {book['loan_status']}"
    )


def find_books(query: str) -> List[dict]:
    """제목이 있으면 그 책만, 작가 이름이면 그 작가의 수록 도서를 모두 찾는다."""
    q = compact(query)
    if not q:
        return []
    titled = title_matches(query)
    if titled:
        return titled
    authors = []
    for book in books:
        name = book["author"]
        if name and matches_term(query, name) and name not in authors:
            authors.append(name)
    if authors:
        return [book for book in books if book["author"] in authors]
    found = []
    for book in books:
        parts = [book["publisher"], book.get("genre", ""), *book["keywords"], *book.get("tags", [])]
        if any(part and matches_term(query, part) for part in parts):
            found.append(book)
    return found


def matches_term(query: str, term: str) -> bool:
    """단어와 조사까지 비교한다. '별'이 '별나라책ZZZ'에 포함됐다고 찾지 않는다."""
    pattern = r"\s*".join(re.escape(ch) for ch in compact(term))
    suffix = r"(?=$|[\s,.?!·/\"'《》]|(?:은|는|이|가|을|를|와|과|의|도|랑|이랑)(?=$|\s|검색|찾아|추천|예약|반납|대출)|검색|찾아|추천|예약|반납|대출)"
    return bool(pattern and re.search(r"(?<![\w])" + pattern + suffix, query.lower()))


def title_matches(query: str) -> List[dict]:
    """질문에 명시된 소장 도서 제목만 찾는다. 주제·작가 검색과 구분한다."""
    return [book for book in books if matches_term(query, book["title"])]

