import requests
import time
import random
import re
import json
import os
from collections import defaultdict

from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
import faiss

KING_MAP = {
    "kaa": "태조", "kba": "정종", "kca": "태종", "kda": "세종", "kea": "문종",
    "kfa": "단종", "kga": "세조", "kha": "예종", "kia": "성종", "kja": "연산군",
    "kka": "중종", "kla": "인종", "kma": "명종", "kna": "선조", "knb": "선조",
    "koa": "광해군", "kob": "광해군", "kpa": "인조", "kqa": "효종", "kra": "현종",
    "krb": "현종", "ksa": "숙종", "ksb": "숙종", "kta": "경종", "ktb": "경종",
    "kua": "영조", "kva": "정조", "kwa": "순조", "kxa": "헌종", "kya": "철종",
    "kza": "고종", "kzb": "순종", "kzc": "순종"
}

KING_START_YEAR = {
    "태조": 1391, "정종": 1398, "태종": 1400, "세종": 1418, "문종": 1450,
    "단종": 1452, "세조": 1454, "예종": 1468, "성종": 1469, "연산군": 1494,
    "중종": 1505, "인종": 1544, "명종": 1545, "선조": 1567, "광해군": 1608,
    "인조": 1622, "효종": 1649, "현종": 1659, "숙종": 1674, "경종": 1720,
    "영조": 1724, "정조": 1776, "순조": 1800, "헌종": 1834, "철종": 1849,
    "고종": 1863, "순종": 1907
}

# -----------------------------------------------
# 브라우저처럼 보이기 위한 헤더 설정
# -----------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://sillok.history.go.kr/",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# -----------------------------------------------
# 세션 + 재시도 전략 설정
# -----------------------------------------------
session = requests.Session()

retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# -----------------------------------------------
# 크롤링 함수 정의
# -----------------------------------------------
def crawl_with_requests(url):
  time.sleep(random.uniform(1.5, 3.5))
  response = session.get(url, headers=HEADERS, timeout=30, verify=True)
  response.raise_for_status()
  response.encoding = "utf-8"
  return response.text

def extract_raw_text_from_html(html_content):
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    selectors = [
        ("div", {"id": "content"}),
        ("article", {}),
        ("main", {}),
        ("body", {}),
    ]

    raw_text = ""

    for tag, attrs in selectors:
        el = soup.find(tag, attrs)
        if el:
            import copy
            target_el = copy.copy(el)

            for unwanted in target_el.find_all(["script", "style", "nav", "header", "footer"]):
                unwanted.decompose()

            text = target_el.get_text(separator="\n", strip=True)

            if len(text) > 100:
                raw_text = text
                break

    return raw_text


def extract_title_from_text(raw_text):
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

    for i, line in enumerate(lines):
        if "신역 바로가기" in line:
            if i > 0:
                return lines[i-1]

    for i, line in enumerate(lines):
        if re.search(r'\d{4}년.*년$', line):
            if i + 1 < len(lines):
                return lines[i+1]

    keywords = ["원본", "책갈피", "국역", "원문", "모두"]
    for i, line in enumerate(lines):
        if any(kw == line for kw in keywords):
            for j in range(i-1, -1, -1):
                if "실록" not in lines[j] and "기사" not in lines[j] and "/" not in lines[j]:
                    return lines[j]

    return "제목 없음"


def extract_content_from_text(raw_text):
    parts = raw_text.split("국역")
    if len(parts) < 2:
        return "국역 컨텐츠를 찾을 수 없습니다."

    target_content = max(parts, key=len)

    kukyeok_body = ""
    copyright_mark = "ⓒ 세종대왕기념사업회"

    if copyright_mark in target_content:
        kukyeok_body = target_content.split(copyright_mark)[0] + copyright_mark
    else:
        if "\n원문" in target_content:
            kukyeok_body = target_content.split("\n원문")[0]
        else:
            kukyeok_body = target_content.split("원문")[0]

    kukyeok_body = kukyeok_body.strip()

    if copyright_mark in raw_text and copyright_mark not in kukyeok_body:
        parts_for_copyright = raw_text.split(copyright_mark)
        if len(parts_for_copyright) > 1:
            potential_body = parts_for_copyright[0] + copyright_mark
            if "국역" in potential_body:
                kukyeok_body = potential_body.split("국역")[-1].strip()

    return kukyeok_body


def extract_category_from_text(raw_text):
    if "【분류】" not in raw_text:
        return []

    category_part = raw_text.split("【분류】")[-1].strip()

    stop_before = ["원본", "첫 페이지", "이전 페이지", "다음 페이지", "마지막 페이지", "일별 목록"]

    lines = category_part.split('\n')
    valid_lines = []
    for line in lines:
        clean_line = line.strip()
        if not clean_line: continue
        if any(stop in clean_line for stop in stop_before):
            break
        valid_lines.append(clean_line)

    combined_text = " ".join(valid_lines)
    items = [item.strip() for item in combined_text.split('/') if item.strip()]

    categories = []
    for item in items:
        clean_item = re.sub(r'\([^)]*\)', '', item).strip()
        if "-" in clean_item and not clean_item.isdigit():
            categories.append(clean_item)

    return list(dict.fromkeys(categories))


def create_sillok_documents(url, raw_text):
    article_id = url.split('/')[-1]
    match = re.search(r'([a-z]{3})_(\d)(\d{2})(\d{2})(\d)(\d{2})_(\d{3})', article_id)

    if not match:
        return []

    k_code, _, y, m, lunar, d, idx = match.groups()

    king = KING_MAP.get(k_code, "알 수 없음")

    y_int = int(y)
    start_year = KING_START_YEAR.get(king, 0)
    solar_year = start_year + y_int

    year_label = f"{y_int}년"
    month_prefix = "윤" if lunar == '1' else ""
    month = f"{month_prefix}{int(m)}월"
    day = f"{int(d)}일"
    article_idx = int(idx)

    title = extract_title_from_text(raw_text)
    content = extract_content_from_text(raw_text)
    categories = extract_category_from_text(raw_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(content)

    docs = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=f"기사 제목: {title}\n날짜: {king} {year_label}({solar_year}년) {month} {day}\n카테고리: {', '.join(categories)}\n본문 내용: {chunk}",
            metadata={
                "type": "article",
                "title": title,
                "king": king,
                "year": year_label,
                "solar_year": solar_year,
                "month": month,
                "day": day,
                "idx": article_idx,
                "chunk_id": i
            }
        )
        docs.append(doc)

    return docs


def collect_sillok_data(url_list):
    all_docs = []
    for url in url_list:
        print(f"🔗 크롤링 중: {url}")
        try:
            html = crawl_with_requests(url)
            raw_text = extract_raw_text_from_html(html)
            docs = create_sillok_documents(url, raw_text)
            all_docs.extend(docs)
        except Exception as e:
            print(f"❌ 예외 발생 ({url}): {e}")
    return all_docs


def save_docs_to_jsonl(docs, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for doc in docs:
            data = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"💾 {len(docs)}개의 문서가 {filename}에 저장되었습니다.")


# -----------------------------------------------
# 월별 그룹화 + 일괄 저장 로직
# -----------------------------------------------
def group_urls_by_month(url_list):
    """
    URL 리스트를 '왕코드_YYMMl' 단위(예: kfa_100050)로 그룹화합니다.
    윤달은 lunar 자리(6번째 숫자)가 1이므로 자연스럽게 별도 그룹으로 분리됩니다.
    dict 삽입 순서가 유지되므로 입력 순서대로 월 그룹이 정렬됩니다.
    """
    groups = defaultdict(list)
    for url in url_list:
        article_id = url.rstrip('/').split('/')[-1]
        # 예: kfa_10005014_001 → 왕코드 kfa, 날짜코드 10005014
        # 앞 6자리 '100050' = 1(prefix) + 00(년) + 05(월) + 0(평/윤)
        m = re.match(r'([a-z]{3})_(\d{6})\d{2}_\d{3}$', article_id)
        if not m:
            print(f"⚠️ 형식이 맞지 않는 URL 건너뜀: {url}")
            continue
        k_code, month_key = m.groups()
        group_key = f"{k_code}_{month_key}"
        groups[group_key].append(url)
    return groups


def crawl_urls_from_file(url_file, output_dir="output", skip_existing=True):
    """
    url_file의 모든 URL을 월별로 그룹화하여
    {왕코드}_{YYMMl}.jsonl 형태로 output_dir에 저장합니다.

    skip_existing=True이면 이미 존재하는 jsonl 파일은 건너뜁니다.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(url_file, 'r', encoding='utf-8') as f:
        url_list = [line.strip() for line in f if line.strip()]

    print(f"📄 총 {len(url_list)}개의 URL을 읽었습니다.")

    groups = group_urls_by_month(url_list)
    print(f"📦 {len(groups)}개의 월 그룹으로 분류되었습니다.\n")

    for i, (group_key, urls) in enumerate(groups.items(), start=1):
        filename = os.path.join(output_dir, f"{group_key}.jsonl")

        if skip_existing and os.path.exists(filename):
            print(f"[{i}/{len(groups)}] ⏭️  이미 존재하여 건너뜀: {filename}")
            continue

        print(f"[{i}/{len(groups)}] ▶️  {group_key} 처리 시작 ({len(urls)}개 기사)")
        docs = collect_sillok_data(urls)
        save_docs_to_jsonl(docs, filename)
        print()


if __name__ == "__main__":
    crawl_urls_from_file("url/문종_url.txt", output_dir="article", skip_existing=True)