"""
retriever.py
────────────
두 가지 검색 전략을 결합한 RAG 검색 엔진.

[거시적 질문] "세종의 생애를 알려줘"
  → Wikipedia 요약 + 실록 대표 기사(연도별 1건씩) 반환

[미시적 질문] "1446년 9월에 훈민정음 반포 때 무슨 일이 있었나요"
  → (1) 질문에서 날짜/키워드 추출
  → (2) 날짜 필터 우선, 이후 벡터 유사도로 top-k 반환
  → (3) 부족할 경우 실록 사이트 실시간 크롤링으로 보완

인덱스 구조:
  ./index/{king_name}.faiss   ← 왕별 분리 인덱스
  ./index/{king_name}_meta.pkl
"""

import re
import pickle
import json
import time
import random
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# crawl.py의 함수 재사용
from crawl import (
    crawl_with_requests,
    extract_raw_text_from_html,
    create_sillok_documents,
    KING_MAP,
    KING_START_YEAR,
)

EMBED_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"
INDEX_DIR     = Path("./index")
DATA_DIR      = Path("./data")
TOP_K_DENSE   = 7    # 벡터 검색 결과 수
TOP_K_MACRO   = 5    # 거시 질문 대표 기사 수 (연도별)

# 왕 이름 → king 코드 매핑 (crawl.py KING_MAP 역방향)
NAME_TO_CODES: dict[str, list[str]] = {}
for code, name in KING_MAP.items():
    NAME_TO_CODES.setdefault(name, []).append(code)


def _index_key(king_name: str) -> str:
    """
    인덱스 파일명에 쓸 ASCII 키를 반환한다.
    한글 파일명은 Windows에서 인코딩 오류를 일으키므로
    대표 king_code(예: 세종 → kda)를 파일명으로 사용한다.
    코드가 없는 비왕 인물은 이름을 그대로 사용(영문/숫자만 허용).
    """
    codes = NAME_TO_CODES.get(king_name)
    if codes:
        return codes[0]                          # 예: "kda"
    # 비왕 인물: 한글 제거 후 영숫자만 남김
    ascii_key = re.sub(r"[^a-zA-Z0-9_]", "_", king_name)
    return ascii_key or "unknown"


# ── 임베딩 모델 (싱글톤) ─────────────────────────────────────────────────────
_embed_model: Optional[SentenceTransformer] = None

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


# ── 인덱스 빌드 & 로드 ───────────────────────────────────────────────────────

def _records_from_data(king_name: str) -> list[dict]:
    """data/ 폴더의 JSON 파일에서 해당 왕의 레코드를 읽는다."""
    records = []
    for path in sorted(DATA_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # crawl.py 형식 (page_content + metadata)
                if "page_content" in obj:
                    if obj.get("metadata", {}).get("king") == king_name:
                        records.append(obj)
                # batch_convert.py 형식 (title + metadata)
                elif obj.get("metadata", {}).get("king") == king_name:
                    meta = obj["metadata"]
                    records.append({
                        "page_content": (
                            f"기사 제목: {obj.get('title','')}\n"
                            f"카테고리: {' '.join(meta.get('class', []))}\n"
                            f"본문 내용: {obj.get('title','')}"
                        ),
                        "metadata": {
                            "king":       meta.get("king"),
                            "year":       meta.get("year"),
                            "solar_year": meta.get("solar_year"),
                            "month":      meta.get("month"),
                            "day":        meta.get("day"),
                            "idx":        meta.get("idx"),
                            "title":      obj.get("title"),
                            "article_id": meta.get("article_id"),
                            "category":   meta.get("class", []),
                            "chunk_id":   0,
                        },
                    })
    return records


def build_king_index(king_name: str) -> tuple[faiss.Index, list[dict]]:
    """왕 이름으로 FAISS 인덱스를 빌드하고 저장한다."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    model   = get_embed_model()
    records = _records_from_data(king_name)
    key     = _index_key(king_name)   # ASCII 파일명 키

    if not records:
        print(f"  ⚠️  {king_name} 데이터 없음 → 빈 인덱스 생성")
        dim   = 384
        index = faiss.IndexFlatIP(dim)
        return index, []

    texts = [r["page_content"] for r in records]
    print(f"  임베딩 중… {king_name} ({len(texts):,}건)")
    vecs = model.encode(texts, batch_size=128, show_progress_bar=False,
                        normalize_embeddings=True).astype("float32")

    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    # 파일명에 한글 대신 ASCII king_code 사용 (Windows 인코딩 오류 방지)
    faiss.write_index(index, str(INDEX_DIR / f"{key}.faiss"))
    with open(INDEX_DIR / f"{key}_meta.pkl", "wb") as f:
        pickle.dump(records, f)

    print(f"  ✅ {king_name} 인덱스 저장 완료 ({key}.faiss)")
    return index, records


def load_king_index(king_name: str) -> tuple[faiss.Index | None, list[dict]]:
    """저장된 왕 인덱스를 로드한다. 없으면 (None, []) 반환."""
    key       = _index_key(king_name)
    idx_path  = INDEX_DIR / f"{key}.faiss"
    meta_path = INDEX_DIR / f"{key}_meta.pkl"
    if not idx_path.exists() or not meta_path.exists():
        return None, []
    index = faiss.read_index(str(idx_path))
    with open(meta_path, "rb") as f:
        records = pickle.load(f)
    return index, records


def get_king_index(king_name: str) -> tuple[faiss.Index, list[dict]]:
    """인덱스가 있으면 로드, 없으면 빌드한다."""
    index, records = load_king_index(king_name)
    if index is None:
        index, records = build_king_index(king_name)
    return index, records


# ── 질문 분류 ────────────────────────────────────────────────────────────────

DATE_PATTERN = re.compile(
    r"(?:(\d{1,4})년)?\s*(?:(윤)?(\d{1,2})월)?\s*(?:(\d{1,2})일)?"
)

def classify_query(query: str) -> dict:
    """
    질문을 분석해 거시/미시 여부와 날짜 정보를 추출한다.

    반환:
      {
        "is_macro": bool,       # True = 거시(생애 전체), False = 미시(특정 시점)
        "year":  int | None,    # 재위 연도 (예: 28 → 세종 28년)
        "month": int | None,
        "day":   int | None,
        "is_lunar": bool,       # 윤달 여부
        "keywords": list[str],  # 핵심 키워드
      }
    """
    macro_hints = ["생애", "업적", "전반", "전체", "소개", "알려줘", "설명해", "어떤 사람", "어떤 왕"]
    is_macro = any(h in query for h in macro_hints)

    # 날짜 추출
    year = month = day = None
    is_lunar = False

    for m in DATE_PATTERN.finditer(query):
        if m.group(1):
            year = int(m.group(1))
        if m.group(2):
            is_lunar = True
        if m.group(3):
            month = int(m.group(3))
        if m.group(4):
            day = int(m.group(4))

    # 날짜가 구체적이면 미시로 강제
    if year or month or day:
        is_macro = False

    # 키워드: 조사·불용어 제거 후 명사 추출 (단순 휴리스틱)
    stop = {"을", "를", "이", "가", "은", "는", "에", "의", "에서", "로", "으로",
            "했", "하셨", "나요", "인가요", "입니까", "알려주세요", "설명해주세요"}
    tokens = re.findall(r"[가-힣]{2,}", query)
    keywords = [t for t in tokens if t not in stop]

    return {
        "is_macro":  is_macro,
        "year":      year,
        "month":     month,
        "day":       day,
        "is_lunar":  is_lunar,
        "keywords":  keywords,
    }


# ── 실록 크롤링 보완 ─────────────────────────────────────────────────────────

def crawl_sillok_date(king_name: str, year: int, month: int,
                      is_lunar: bool = False, day: int | None = None) -> list[dict]:
    """
    특정 날짜의 실록 기사를 실시간 크롤링해서 반환.
    day가 None이면 해당 월 전체를 수집.
    """
    codes = NAME_TO_CODES.get(king_name, [])
    if not codes:
        return []

    docs_all = []
    days_range = [day] if day else range(1, 32)
    lunar_str  = "1" if is_lunar else "0"

    for k_code in codes:
        year_str  = f"{year:02d}"
        month_str = f"{month:02d}"
        for d_int in days_range:
            day_str = f"{d_int:02d}"
            for i_int in range(1, 30):
                idx_str = f"{i_int:03d}"
                url = (f"https://sillok.history.go.kr/id/"
                       f"{k_code}_1{year_str}{month_str}{lunar_str}{day_str}_{idx_str}")
                try:
                    html     = crawl_with_requests(url)
                    raw_text = extract_raw_text_from_html(html)
                    docs     = create_sillok_documents(url, raw_text)
                    if docs:
                        for doc in docs:
                            docs_all.append({
                                "page_content": doc.page_content,
                                "metadata":     doc.metadata,
                            })
                    else:
                        break   # 해당 날짜 기사 끝
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 404:
                        break
                    break
                except Exception:
                    break

    return docs_all


# ── 핵심 검색 함수 ───────────────────────────────────────────────────────────

def retrieve(
    query: str,
    king_name: str,
    index: faiss.Index,
    records: list[dict],
    active_years: tuple[int, int],
    persona_summary: str,
) -> tuple[list[dict], dict]:
    """
    질문 분류 → 거시/미시 분기 → 최종 컨텍스트 청크 반환.

    반환:
      (chunks, query_info)
      chunks: [{"page_content": ..., "metadata": {...}}, ...]
    """
    qinfo  = classify_query(query)
    model  = get_embed_model()
    chunks = []

    # ── 거시적 질문 ──────────────────────────────────────────────────────
    if qinfo["is_macro"]:
        # 1) 연도별 대표 기사 1건씩 (최대 TOP_K_MACRO년 분)
        seen_years: set[int] = set()
        for rec in records:
            sy = rec["metadata"].get("solar_year")
            if sy and sy not in seen_years:
                seen_years.add(sy)
                chunks.append(rec)
            if len(chunks) >= TOP_K_MACRO * 4:
                break

        # 2) 추가로 벡터 유사도 top-k
        if index.ntotal > 0:
            qvec = model.encode([query], normalize_embeddings=True).astype("float32")
            _, I = index.search(qvec, TOP_K_DENSE)
            for i in I[0]:
                if i >= 0 and records[i] not in chunks:
                    chunks.append(records[i])

        return chunks[:TOP_K_DENSE * 2], qinfo

    # ── 미시적 질문 ──────────────────────────────────────────────────────

    # 1단계: 날짜 필터 (인덱스 내 레코드에서)
    filtered = records
    if qinfo["year"]:
        # 재위 연도를 서기 연도로 변환
        start_year = active_years[0]
        target_solar = start_year + qinfo["year"] - 1  # 1년 = 즉위년+0
        filtered = [r for r in filtered
                    if r["metadata"].get("solar_year") == target_solar]
    if qinfo["month"]:
        m_prefix = "윤" if qinfo["is_lunar"] else ""
        target_m = f"{m_prefix}{qinfo['month']}월"
        filtered = [r for r in filtered
                    if r["metadata"].get("month") == target_m]
    if qinfo["day"]:
        target_d = f"{qinfo['day']}일"
        filtered = [r for r in filtered
                    if r["metadata"].get("day") == target_d]

    # 2단계: 날짜 필터 결과 내에서 벡터 유사도 재정렬
    if filtered and index.ntotal > 0:
        texts = [r["page_content"] for r in filtered]
        qvec  = model.encode([query], normalize_embeddings=True).astype("float32")
        fvecs = model.encode(texts, batch_size=64, normalize_embeddings=True
                             ).astype("float32")
        scores = (fvecs @ qvec.T).squeeze()
        if scores.ndim == 0:
            scores = np.array([float(scores)])
        order  = np.argsort(scores)[::-1][:TOP_K_DENSE]
        chunks = [filtered[i] for i in order]

    # 3단계: 결과가 부족하면 실시간 크롤링으로 보완
    if len(chunks) < 3 and qinfo["year"] and qinfo["month"]:
        print(f"\n  🌐 실록 사이트 실시간 크롤링 중 "
              f"({king_name} {qinfo['year']}년 {qinfo['month']}월) …")
        live_docs = crawl_sillok_date(
            king_name,
            year=qinfo["year"],
            month=qinfo["month"],
            is_lunar=qinfo["is_lunar"],
            day=qinfo["day"],
        )
        # 크롤링 결과도 벡터 유사도로 재정렬
        if live_docs:
            texts  = [d["page_content"] for d in live_docs]
            qvec   = model.encode([query], normalize_embeddings=True).astype("float32")
            fvecs  = model.encode(texts, batch_size=64, normalize_embeddings=True
                                  ).astype("float32")
            scores = (fvecs @ qvec.T).squeeze()
            if scores.ndim == 0:
                scores = np.array([float(scores)])
            order  = np.argsort(scores)[::-1][:TOP_K_DENSE]
            chunks = [live_docs[i] for i in order]

    # 4단계: 여전히 비어 있으면 전체 인덱스 벡터 검색
    if not chunks and index.ntotal > 0:
        qvec = model.encode([query], normalize_embeddings=True).astype("float32")
        _, I = index.search(qvec, TOP_K_DENSE)
        chunks = [records[i] for i in I[0] if i >= 0]

    return chunks, qinfo