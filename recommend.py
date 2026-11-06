"""도서 추천.

1) 같은 작가는 빼고  2) 다른 독자의 다음 대출·평점·급상승으로 순서를 매긴 뒤
3) 키워드·태그·줄거리가 비슷한 정도를 보조 점수로 쓴다.
"""

from typing import Dict, List, Optional

from catalog import book_by_title, books


def cosine_similarity(a, b):
    """03_rag.py와 같은 계산. 1에 가까울수록 키워드가 비슷하다."""
    dot = sum(x * y for x, y in zip(a, b))
    left = sum(x * x for x in a) ** 0.5
    right = sum(y * y for y in b) ** 0.5
    if left == 0 or right == 0:
        return 0.0
    return dot / (left * right)

# 기준 도서를 읽은 뒤 다른 독자가 이어서 빌린 횟수와 평점
also_borrowed: Dict[str, List[Dict]] = {
    "시간을 파는 서점": [
        {"title": "닫힌 서점의 내일", "next_loans": 28, "avg_rating": 4.9},
        {"title": "고양이는 자정에 책을 읽는다", "next_loans": 21, "avg_rating": 4.6},
        {"title": "마법사가 사라진 왕국", "next_loans": 17, "avg_rating": 4.2},
        {"title": "구름 위 우체국의 마지막 편지", "next_loans": 9, "avg_rating": 4.4},
    ],
    "고양이는 자정에 책을 읽는다": [
        {"title": "골목의 고양이 시계", "next_loans": 26, "avg_rating": 4.8},
        {"title": "시간을 파는 서점", "next_loans": 19, "avg_rating": 4.7},
        {"title": "구름 위 우체국의 마지막 편지", "next_loans": 11, "avg_rating": 4.5},
    ],
    "달빛 도서관의 비밀": [
        {"title": "기억을 잃은 탐정의 편지", "next_loans": 14, "avg_rating": 4.4},
        {"title": "열세 번째 문이 열리는 밤", "next_loans": 12, "avg_rating": 4.1},
    ],
}

# 최근 대출, 베스트셀러 순위, 급상승 여부
trends: Dict[str, Dict] = {
    "달빛 도서관의 비밀": {"recent_loans": 18, "bestseller_rank": 5, "rising": False},
    "내일이 사라진 도시": {"recent_loans": 22, "bestseller_rank": 4, "rising": True},
    "고양이는 자정에 책을 읽는다": {"recent_loans": 36, "bestseller_rank": 3, "rising": True},
    "시간을 파는 서점": {"recent_loans": 20, "bestseller_rank": 7, "rising": False},
    "여름 끝에서 너를 만났다": {"recent_loans": 12, "bestseller_rank": 8, "rising": False},
    "기억을 잃은 탐정의 편지": {"recent_loans": 9, "bestseller_rank": 9, "rising": False},
    "열세 번째 문이 열리는 밤": {"recent_loans": 11, "bestseller_rank": 10, "rising": False},
    "구름 위 우체국의 마지막 편지": {"recent_loans": 14, "bestseller_rank": 6, "rising": False},
    "마법사가 사라진 왕국": {"recent_loans": 48, "bestseller_rank": 1, "rising": True},
    "우리의 비밀은 별빛보다 깊다": {"recent_loans": 8, "bestseller_rank": 11, "rising": False},
    "사과의 매력을 찾아서": {"recent_loans": 5, "bestseller_rank": 13, "rising": False},
    "닫힌 서점의 내일": {"recent_loans": 50, "bestseller_rank": 2, "rising": True},
    "골목의 고양이 시계": {"recent_loans": 33, "bestseller_rank": 12, "rising": True},
}
EMPTY_TREND = {"recent_loans": 0, "bestseller_rank": 99, "rising": False}


def _flag(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() not in ("0", "false", "no", "아니오", "꺼")


def _tokens(book: dict) -> List[str]:
    words = []
    words.extend(book.get("keywords", []))
    words.extend(book.get("tags", []))
    words.extend(str(book.get("synopsis", "")).split())
    cleaned = []
    for word in words:
        if len(word) >= 2 and word not in cleaned:
            cleaned.append(word)
    return cleaned


def _vibe(left: dict, right: dict):
    """두 책의 키워드 개수로 벡터를 만들어 코사인 유사도를 구한다."""
    left_words = _tokens(left)
    right_words = _tokens(right)
    vocab = []
    for word in left_words + right_words:
        if word not in vocab:
            vocab.append(word)
    if not vocab:
        return 0.0, []
    left_vec = [left_words.count(word) for word in vocab]
    right_vec = [right_words.count(word) for word in vocab]
    shared = []
    for word in vocab:
        if word in left_words and word in right_words:
            shared.append(word)
    return cosine_similarity(left_vec, right_vec), shared


def rank_recommendations(
    anchor_titles: List[str],
    already_read: Optional[List[str]] = None,
    exclude_same_author: bool = True,
    use_trends: bool = True,
    use_vibe: bool = True,
    limit: int = 3,
) -> str:
    """추천 문장을 만든다. 같은 작가 제외가 1순위다."""
    already_read = already_read or []
    exclude_same_author = _flag(exclude_same_author, True)
    use_trends = _flag(use_trends, True)
    use_vibe = _flag(use_vibe, True)
    anchors = []
    for title in anchor_titles:
        book = book_by_title(title)
        if book:
            anchors.append(book)
    if not anchors:
        return "기준이 되는 도서를 찾지 못했습니다. 읽은 책 제목을 알려 주세요."
    if not use_trends and not use_vibe:
        return "독자 동향 또는 감성 유사도 중 하나는 켜져 있어야 추천할 수 있습니다."

    blocked_titles = []
    blocked_authors = []
    for book in anchors:
        blocked_titles.append(book["title"])
        if book["author"] not in blocked_authors:
            blocked_authors.append(book["author"])
    for title in already_read:
        if title not in blocked_titles:
            blocked_titles.append(title)
        book = book_by_title(title)
        if book and book["author"] not in blocked_authors:
            blocked_authors.append(book["author"])
    max_loans = max(item["recent_loans"] for item in trends.values()) or 1
    rank_count = len(trends) or 1
    rows = []
    excluded = []
    for book in books:
        if book["title"] in blocked_titles:
            continue
        if exclude_same_author and book["author"] in blocked_authors:
            excluded.append(book)
            continue
        collab = 0.0
        next_loans = 0
        avg_rating = 0.0
        trend_row = trends.get(book["title"], EMPTY_TREND)
        if use_trends:
            for anchor in anchors:
                for item in also_borrowed.get(anchor["title"], []):
                    if item["title"] == book["title"]:
                        next_loans += item["next_loans"]
                        avg_rating = item["avg_rating"]
                        collab += item["next_loans"] * (item["avg_rating"] / 5)
            trend = trend_row["recent_loans"] / max_loans
            trend += (rank_count - trend_row["bestseller_rank"] + 1) / rank_count
            if trend_row["rising"]:
                trend += 0.3
        else:
            trend = 0.0
        vibe, shared = (0.0, [])
        if use_vibe:
            vibe, shared = max((_vibe(anchor, book) for anchor in anchors), key=lambda pair: pair[0])
        rows.append(
            {
                "book": book,
                "primary": collab + trend if use_trends else 0.0,
                "vibe": vibe,
                "shared": shared,
                "next_loans": next_loans,
                "avg_rating": avg_rating,
                "trend": trend_row,
            }
        )
    rows.sort(key=lambda row: (row["primary"], row["vibe"]), reverse=True)
    picked = [row for row in rows if row["primary"] > 0 or row["vibe"] > 0][:limit]
    if not picked:
        return "같은 작가를 뺀 뒤 동향이나 감성이 맞는 다른 책이 없습니다."

    lines = [
        "기준 도서: " + ", ".join(f"{book['title']} / {book['author']}" for book in anchors),
        f"필터: 같은 작가 제외={exclude_same_author}, 독자 동향={use_trends}, 감성 유사={use_vibe}",
    ]
    if exclude_same_author and excluded:
        lines.append("제외: " + ", ".join(f"{book['title']}({book['author']})" for book in excluded))
    lines.append("추천:")
    for index, row in enumerate(picked, start=1):
        book = row["book"]
        trend = row["trend"]
        bits = []
        if use_trends:
            if row["next_loans"]:
                bits.append(f"이어 대출 {row['next_loans']}명, 평점 {row['avg_rating']:.1f}")
            else:
                bits.append("이어 대출 기록 없음")
            rising = "급상승" if trend["rising"] else "유지"
            bits.append(f"최근 대출 {trend['recent_loans']}권, 베스트 {trend['bestseller_rank']}위, {rising}")
        if use_vibe:
            shared = ", ".join(row["shared"][:4]) if row["shared"] else "없음"
            bits.append(f"감성 유사 {row['vibe']:.2f} ({shared})")
        lines.append(f"{index}. {book['title']} / {book['author']} / " + ", ".join(bits))
    return "\n".join(lines)
