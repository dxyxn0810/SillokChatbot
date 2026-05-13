import requests
import time
import random
import re
import json

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
    total=5,                          # 최대 재시도 횟수
    backoff_factor=2,                 # 대기 시간 배수 (2, 4, 8, 16초...)
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
  time.sleep(random.uniform(1.5, 3.5))  # 사람처럼 보이도록 랜덤 딜레이
  response = session.get(url, headers=HEADERS, timeout=30, verify=True)
  response.raise_for_status()
  response.encoding = "utf-8"
  return response.text

def extract_raw_text_from_html(html_content):
    """
    HTML 콘텐츠에서 불필요한 태그를 제거하고 유의미한 본문 텍스트만 추출합니다.
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    # 텍스트를 추출할 후보 선택자 (중요도 순)
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
            # 원본 보존을 위해 엘리먼트 복제 (선택 사항)
            import copy
            target_el = copy.copy(el)

            # 1. 불필요한 태그(스크립트, 스타일, 네비게이션 등) 제거
            for unwanted in target_el.find_all(["script", "style", "nav", "header", "footer"]):
                unwanted.decompose()

            # 2. 텍스트 추출 (줄바꿈 구분자 유지)
            text = target_el.get_text(separator="\n", strip=True)

            # 3. 추출된 텍스트가 유의미한 길이(100자 이상)인지 확인
            if len(text) > 100:
                raw_text = text
                break

    return raw_text


def extract_title_from_text(raw_text):
    """
    조선왕조실록 raw_text에서 기사 제목을 추출합니다.
    """
    # 1. 줄바꿈을 기준으로 텍스트를 나눕니다.
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    
    # 2. '신역 바로가기' 키워드를 찾습니다. 
    # 보통 제목은 이 키워드 바로 윗줄에 위치합니다.
    for i, line in enumerate(lines):
        if "신역 바로가기" in line:
            if i > 0:
                return lines[i-1]
                
    # 3. 만약 위 로직으로 못 찾았을 경우, 연도 표시(홍무, 영락 등) 다음 줄을 찾습니다.
    # 패턴: 4자리 숫자 + 년 + ... + 년
    for i, line in enumerate(lines):
        if re.search(r'\d{4}년.*년$', line):
            if i + 1 < len(lines):
                return lines[i+1]

    # 4. 최후의 수단: 특정 키워드(원본, 책갈피, 국역)들이 몰려있는 구간 직전의 유의미한 줄
    keywords = ["원본", "책갈피", "국역", "원문", "모두"]
    for i, line in enumerate(lines):
        if any(kw == line for kw in keywords):
            # 키워드들보다 위에 있는 줄 중 '기사'나 '태조실록'이 아닌 첫 번째 줄
            for j in range(i-1, -1, -1):
                if "실록" not in lines[j] and "기사" not in lines[j] and "/" not in lines[j]:
                    return lines[j]

    return "제목 없음"

# 테스트 (ex1 기준)
# 결과: "은언군의 신유년의 죽음을 변무하는 주문"

def extract_content_from_text(raw_text):
    """
    조선왕조실록 raw_text에서 국역 본문, 서지정보, 분류, 저작권을 정밀하게 추출합니다.
    """
    # 1. '국역' 단어로 분할하여 가장 내용이 많은(본문일 가능성이 높은) 섹션 선택
    parts = raw_text.split("국역")
    if len(parts) < 2:
        return "국역 컨텐츠를 찾을 수 없습니다."
    
    # 여러 '국역' 섹션 중 가장 긴 것을 후보로 선택 (메뉴 텍스트 방지)
    target_content = max(parts, key=len)

    # 2. '원문' 키워드 처리 (본문 내의 '원문(願文)'과 섹션 구분자 '원문'을 구분)
    # 한문 원문 섹션은 보통 줄바꿈 후에 '원문'이라는 단어만 단독으로 있거나 
    # 특정 패턴으로 시작하므로, 이를 기준으로 자릅니다.
    
    # 단순히 .split("원문")을 하지 않고, 뒤쪽에서부터 한문이 시작되는 지점을 찾습니다.
    # 실록 구조상 'ⓒ 세종대왕기념사업회'가 국역의 끝임을 이용합니다.
    
    kukyeok_body = ""
    copyright_mark = "ⓒ 세종대왕기념사업회"
    
    if copyright_mark in target_content:
        # 저작권 표시가 있다면 그 지점까지만 자릅니다.
        kukyeok_body = target_content.split(copyright_mark)[0] + copyright_mark
    else:
        # 저작권 표시가 없는 경우, '원문' 섹션이 시작되기 전까지만 가져옵니다.
        # 이때 본문 안의 '원문(願文)'을 보호하기 위해 '\n원문' 패턴을 우선 확인합니다.
        if "\n원문" in target_content:
            kukyeok_body = target_content.split("\n원문")[0]
        else:
            kukyeok_body = target_content.split("원문")[0]

    # 3. 양 끝 정리
    kukyeok_body = kukyeok_body.strip()

    # 4. (보정) 만약 저작권 문구가 원본 raw_text에는 있는데 추출된 body에 없다면 다시 붙여줌
    if copyright_mark in raw_text and copyright_mark not in kukyeok_body:
        # 본문 하단에 서지정보나 분류가 있을 수 있으므로 이를 포함한 범위를 재설정
        # '원문' 섹션 바로 전까지의 텍스트 중 저작권 마크를 포함하도록 함
        parts_for_copyright = raw_text.split(copyright_mark)
        if len(parts_for_copyright) > 1:
            # 첫 번째 저작권 표시가 나오는 곳까지를 본문 영역으로 간주
            # (두 번 반복되는 레이아웃 대응)
            potential_body = parts_for_copyright[0] + copyright_mark
            # 그 중에서 진짜 '국역' 섹션 내용만 필터링
            if "국역" in potential_body:
                kukyeok_body = potential_body.split("국역")[-1].strip()

    return kukyeok_body

# 테스트 실행 (ex1의 경우)
# result = extract_content_from_text(raw_text_ex1)
# print(result)

def extract_category_from_text(raw_text):
    """
    raw_text에서 【분류】 섹션을 찾아 유효한 카테고리만 추출합니다.
    페이지 하단 버튼(이전 페이지, 이동 등)은 필터링합니다.
    """
    if "【분류】" not in raw_text:
        return []

    # 1. '【분류】' 이후 섹션 추출
    category_part = raw_text.split("【분류】")[-1].strip()
    
    # 2. 유의미한 데이터가 끝나는 지점(주석이나 다른 섹션 시작)에서 자르기
    # '원본', '첫 페이지' 등은 버튼이므로 그 직전까지만 데이터로 인정합니다.
    stop_before = ["원본", "첫 페이지", "이전 페이지", "다음 페이지", "마지막 페이지", "일별 목록"]
    
    # 줄 단위로 나누어 유효한 줄만 선택
    lines = category_part.split('\n')
    valid_lines = []
    for line in lines:
        clean_line = line.strip()
        if not clean_line: continue
        # 중단 단어가 나오면 그 이후는 아예 보지 않음
        if any(stop in clean_line for stop in stop_before):
            break
        valid_lines.append(clean_line)
    
    # 다시 합쳐서 '/'로 분리
    combined_text = " ".join(valid_lines)
    items = [item.strip() for item in combined_text.split('/') if item.strip()]
    
    categories = []
    for item in items:
        # 한자 및 괄호 제거
        clean_item = re.sub(r'\([^)]*\)', '', item).strip()
        
        # 3. 최종 필터링 로직
        # - 실록의 카테고리는 항상 '대분류-중분류' 형태를 가집니다 (예: 왕실-의식)
        # - 단순히 '1', '이동' 같은 단어는 대시(-)가 없으므로 제외됩니다.
        if "-" in clean_item and not clean_item.isdigit():
            categories.append(clean_item)
            
    # 중복 제거
    return list(dict.fromkeys(categories))


def create_sillok_documents(url, raw_text):
    """
    URL과 raw_text를 받아 메타데이터(solar_year 포함)를 추출하고 
    LangChain Document 객체 리스트를 생성합니다.
    """
    
    # [Step 1] URL에서 메타데이터 파싱
    article_id = url.split('/')[-1]
    # 정규표현식: 왕코드_1/년2/월2/일2_인덱스3 (예: kaa_10612001_001)
    match = re.search(r'([a-z]{3})_(\d)(\d{2})(\d{2})(\d)(\d{2})_(\d{3})', article_id)

    if not match:
        return []

    k_code, _, y, m, lunar, d, idx = match.groups()
    
    # 왕 이름 추출
    king = KING_MAP.get(k_code, "알 수 없음")
    
    # --- [추가 로직: 서기 연도(solar_year) 계산] ---
    # KING_START_YEAR에서 기준년도를 가져와 재위 년수(y_int)를 더함
    y_int = int(y)
    start_year = KING_START_YEAR.get(king, 0)
    solar_year = start_year + y_int
    # ----------------------------------------------

    year_label = f"{y_int}년"
    month_prefix = "윤" if lunar == '1' else ""
    month = f"{month_prefix}{int(m)}월"
    day = f"{int(d)}일"
    article_idx = int(idx)

    # [Step 2] 데이터 추출
    title = extract_title_from_text(raw_text)
    content = extract_content_from_text(raw_text)
    categories = extract_category_from_text(raw_text)

    # [Step 3] 텍스트 스플리터 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # [Step 4] 본문 쪼개기
    chunks = text_splitter.split_text(content)
    
    # [Step 5] Document 객체화
    # Leaf node document & metadata 생성
    docs = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=f"기사 제목: {title}\n날짜: {king} {year_label}({solar_year}년) {month} {day}\n카테고리: {', '.join(categories)}\n본문 내용: {chunk}",
            metadata={
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

# URL로부터 Document 객체를 생성하는 테스트 코드

"""
URL = "https://sillok.history.go.kr/id/kda_10008011_001"

# 전역 변수 초기화 (이후 단계에서 사용)
html_content = None

print(f"🔗 대상 URL: {URL}")

html_content = crawl_with_requests(URL)
print("✅ HTML 크롤링 완료")
raw_text = extract_raw_text_from_html(html_content)
print("✅ 텍스트 추출 완료")
file_name = "sillok_raw_text.txt"

with open(file_name, "w", encoding="utf-8") as f:
    f.write(raw_text)

print(f"\n✅ 파일 저장 완료: {file_name}")
print(f"📏 저장된 텍스트 길이: {len(raw_text)} 자")
    
print("\n--- 추출된 메타데이터 ---")
print(extract_title_from_text(raw_text))
print(extract_category_from_text(raw_text))
print(extract_content_from_text(raw_text)[:500])  # 본문 앞 500자 미리보기
print("\n--- Document 객체 예시 ---")

docs = create_sillok_documents(URL, raw_text)
for doc in docs:
    print(f"--- Document Chunk {doc.metadata['chunk_id']} ---")
    print(f"{doc.page_content}")
    print("\n")
"""


def collect_sillok_data(url_list):
    """
    주어진 URL 리스트에서 실록 데이터를 수집합니다.
    """
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
            # Document 객체를 딕셔너리로 변환하여 저장
            data = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"💾 {len(docs)}개의 문서가 {filename}에 저장되었습니다.")

# 테스트 코드
# url_list = [
#     "https://sillok.history.go.kr/id/kda_10008011_001",
#     "https://sillok.history.go.kr/id/kda_10008011_002",
#     "https://sillok.history.go.kr/id/kda_10008011_003"
# ]
# docs = collect_sillok_data(url_list)
# save_docs_to_jsonl(docs, "data.jsonl")