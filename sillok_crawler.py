"""
조선왕조실록 전체 기사 URL 수집기
=====================================
전략:
  1. /search/inspectionList.do  → 왕 코드 목록 (kaa, kba, ...) 추출
  2. /search/inspectionMonthList.do?id={king_code} → 해당 왕의 연월 ID 목록 추출
     (search('kaa_10107', ...) 형식)
  3. /search/inspectionView.do  POST {id: month_id} → 해당 월의 모든 기사 URL 추출
     (/id/kaa_10107017_001 형식)

서버 부하 최소화:
  - 요청 간 딜레이 적용 (기본 0.5초)
  - 재시도 로직 포함 (최대 3회)
  - 이미 수집된 결과는 중간 파일에 저장하여 재시작 가능

사용법:
  python sillok_crawler.py                  # 기본 실행 (output.txt에 저장)
  python sillok_crawler.py -o links.txt     # 출력 파일 지정
  python sillok_crawler.py --delay 1.0      # 요청 간 딜레이 1초
  python sillok_crawler.py --resume         # 중간 저장 파일에서 재개
"""

import urllib.request
import urllib.parse
import urllib.error
import re
import time
import argparse
import os
import sys
import json
from datetime import datetime

# ===========================================================================
# 설정
# ===========================================================================
BASE_URL = "https://sillok.history.go.kr"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    "Referer": "https://sillok.history.go.kr/search/inspectionList.do",
}

KING_MAP = {
    "kaa": "태조", "kba": "정종", "kca": "태종", "kda": "세종", "kea": "문종",
    "kfa": "단종", "kga": "세조", "kha": "예종", "kia": "성종", "kja": "연산군",
    "kka": "중종", "kla": "인종", "kma": "명종", "kna": "선조", "knb": "선조",
    "koa": "광해군", "kob": "광해군", "kpa": "인조", "kqa": "효종", "kra": "현종",
    "krb": "현종", "ksa": "숙종", "ksb": "숙종", "kta": "경종", "ktb": "경종",
    "kua": "영조", "kva": "정조", "kwa": "순조", "kxa": "헌종", "kya": "철종",
    "kza": "고종", "kzb": "순종", "kzc": "순종"
}

# ===========================================================================
# HTTP 헬퍼
# ===========================================================================

def fetch_get(path: str, params: dict = None, max_retries: int = 3, delay: float = 0.5) -> str:
    """GET 요청. 실패 시 최대 max_retries회 재시도."""
    url = BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # 404는 재시도 안 함
            print(f"  [HTTP {e.code}] {url} (attempt {attempt}/{max_retries})", file=sys.stderr)
        except Exception as e:
            print(f"  [Error] {url}: {e} (attempt {attempt}/{max_retries})", file=sys.stderr)
        
        if attempt < max_retries:
            time.sleep(delay * attempt * 2)  # 재시도 대기는 더 길게
    
    return None


def fetch_post(path: str, data: dict, max_retries: int = 3, delay: float = 0.5) -> str:
    """POST 요청. 실패 시 최대 max_retries회 재시도."""
    url = BASE_URL + path
    headers = {
        **HEADERS,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data_bytes = urllib.parse.urlencode(data).encode("utf-8")
    
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, data=data_bytes, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            print(f"  [HTTP {e.code}] POST {url} data={data} (attempt {attempt}/{max_retries})", file=sys.stderr)
        except Exception as e:
            print(f"  [Error] POST {url}: {e} (attempt {attempt}/{max_retries})", file=sys.stderr)
        
        if attempt < max_retries:
            time.sleep(delay * attempt * 2)
    
    return None


# ===========================================================================
# 데이터 추출 함수
# ===========================================================================

def get_king_codes() -> list[str]:
    """
    /search/inspectionList.do 에서 왕 코드 목록을 추출.
    예: ['kaa', 'kba', 'kca', ..., 'kza']
    """
    print("왕 코드 목록 수집 중...")
    html = fetch_get("/search/inspectionList.do")
    if not html:
        raise RuntimeError("왕 목록 페이지를 가져오는 데 실패했습니다.")
    
    # <li id="kaa" level="1" ...> 패턴
    codes = re.findall(r'<li\s+id="(k[a-z]{2})"', html)
    codes = list(dict.fromkeys(codes))  # 순서 유지 dedup
    print(f"  발견된 왕 코드: {codes}")
    return codes


def get_month_ids(king_code: str, delay: float) -> list[str]:
    """
    /search/inspectionMonthList.do?id={king_code} 에서 연/월 ID 목록을 추출.
    예: ['kaa_10107', 'kaa_10108', ..., 'kaa_20101', ...]
    형식: kXX_YYYММ  (앞의 1은 고정, YYY=3자리 재위년, MM=2자리 월)
    참고: search('kaa_101070', '1', '2') 형식이지만 마지막 0은 윤달 구분자가 포함된 형태.
    실제로 사용되는 ID는 search()의 첫번째 인자.
    """
    html = fetch_get("/search/inspectionMonthList.do", {"id": king_code})
    if not html:
        print(f"  [경고] {king_code} 월 목록을 가져오지 못했습니다.", file=sys.stderr)
        return []
    
    # search('kaa_10107', '1', '2') — 첫 번째 인자가 연월 ID
    # 6자리 숫자: YYYММ (재위년3자리 + 월2자리) + 선택적 윤달자리(1자리)
    # inspectionView에는 6자리(YYYMM0 형태)가 아닌 5자리(YYYMM)를 써야 한다
    raw_ids = re.findall(r"search\('(k[a-z]{2}_\d+)'", html)
    
    # 중복 제거, 순서 유지
    seen = set()
    month_ids = []
    for mid in raw_ids:
        if mid not in seen and not mid.endswith("_000"):
            seen.add(mid)
            month_ids.append(mid)
    
    return month_ids


def get_article_ids_for_month(month_id: str, delay: float) -> list[str]:
    """
    /search/inspectionView.do POST {id: month_id} 에서 해당 월의 기사 ID 목록을 추출.
    예: ['kaa_10107017_001', 'kaa_10107017_002', ...]
    중복 제거 후 반환.
    """
    html = fetch_post("/search/inspectionView.do", {"id": month_id, "level": "2"})
    if not html:
        return []
    
    # /id/kXX_YYMMDD_NNN 형식 링크 추출
    raw = re.findall(r'/id/(k[a-z]{2}_\d+_\d+)', html)
    
    # 중복 제거, 순서 유지
    seen = set()
    article_ids = []
    for aid in raw:
        if aid not in seen:
            seen.add(aid)
            article_ids.append(aid)
    
    return article_ids


# ===========================================================================
# 진행 상황 저장/로드 (재시작 지원)
# ===========================================================================

PROGRESS_FILE = "sillok_progress.json"

def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_months": [], "articles": []}

def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ===========================================================================
# 메인 크롤러
# ===========================================================================

def crawl(output_file: str, delay: float, resume: bool):
    """
    전체 기사 URL을 수집하여 'url' 폴더 내에 왕별로 txt 파일을 생성합니다.
    """
    start_time = datetime.now()
    
    # 0. 저장 폴더 준비
    target_dir = "url"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"[*] '{target_dir}' 폴더를 생성했습니다.")

    print(f"[{start_time:%Y-%m-%d %H:%M:%S}] 크롤링 시작")
    print(f"요청 딜레이: {delay}초")
    print("-" * 40)

    # 1. 진행 상황 로드 (재개 모드)
    progress = load_progress() if (resume and os.path.exists(PROGRESS_FILE)) else {"completed_months": []}
    completed_months = set(progress.get("completed_months", []))
    
    if resume:
        print(f"[*] 재개 모드: 기존 완료된 월 {len(completed_months)}개를 제외하고 시작합니다.")

    # 2. 왕 코드 목록 수집
    try:
        king_codes = get_king_codes()
    except RuntimeError as e:
        print(f"[오류] {e}")
        return

    time.sleep(delay)
    total_articles_count = 0
    
    start_king_code = "kia"
    start_index = king_codes.index(start_king_code)
    target_king_codes = king_codes[start_index:]

    # 3. 왕별 루프
    for king_idx, king_code in enumerate(target_king_codes, 1):
        king_name = KING_MAP.get(king_code, king_code)
        print(f"\n[{king_idx}/{len(king_codes)}] {king_name}({king_code}) 데이터 수집 중...")
        
        king_articles = []  # 이 왕의 URL들을 임시 저장할 리스트
        
        # 해당 왕의 연월 목록 가져오기
        month_ids = get_month_ids(king_code, delay)
        time.sleep(delay)
        
        for month_idx, month_id in enumerate(month_ids, 1):
            # 이미 처리된 월이면 건너뜀
            if month_id in completed_months:
                # 건너뛰더라도 기존 파일에 이미 있을 것이므로 로그만 출력
                continue
            
            # 해당 월의 모든 기사 ID 가져오기
            article_ids = get_article_ids_for_month(month_id, delay)
            month_urls = [f"{BASE_URL}/id/{aid}" for aid in article_ids]
            
            king_articles.extend(month_urls)
            total_articles_count += len(month_urls)
            
            print(f"  - {month_id}: {len(month_urls)}건 수집 (현재 왕 누적: {len(king_articles)}건)")
            
            # 진행 상태 업데이트 및 저장
            completed_months.add(month_id)
            save_progress({"completed_months": list(completed_months)})
            
            time.sleep(delay)
        
        # 4. 한 왕의 수집이 끝나면 파일로 저장
        if king_articles:
            file_path = os.path.join(target_dir, f"{king_name}_url.txt")
            # 'a' 모드를 사용하여 재개 시에도 기존 데이터 뒤에 붙여넣음
            with open(file_path, "a", encoding="utf-8") as f:
                f.write("\n".join(king_articles) + "\n")
            print(f"  >> [저장 완료] {file_path}")
        else:
            if month_ids: # 월 목록은 있는데 새로 수집된 기사가 없는 경우 (이미 다 완료된 경우 등)
                print(f"  >> {king_name}은(는) 새로 추가된 기사가 없습니다.")

    # 5. 최종 종료 처리
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"작업 완료!")
    print(f"총 수집된 기사 URL: {total_articles_count}개")
    print(f"저장 위치: ./{target_dir}/")
    print(f"총 소요 시간: {elapsed}")
    print(f"{'='*60}")
    
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print(f"[*] 임시 진행 파일({PROGRESS_FILE})을 삭제했습니다.")


# ===========================================================================
# 엔트리포인트
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="조선왕조실록 전체 기사 URL 수집기"
    )
    parser.add_argument(
        "-o", "--output",
        default="sillok_links.txt",
        help="출력 파일 경로 (기본값: sillok_links.txt)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="요청 사이 대기 시간(초). 기본값: 0.5. 서버 차단 우려 시 1.0 이상 권장."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"중단된 작업을 {PROGRESS_FILE}에서 재개"
    )
    args = parser.parse_args()
    
    crawl(
        output_file=args.output,
        delay=args.delay,
        resume=args.resume,
    )
