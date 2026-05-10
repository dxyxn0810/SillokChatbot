"""
조선왕조실록 XML 배치 변환 스크립트

파일 규칙:
  입력  ./xml/2nd_{king_code}_{idx}.xml
  출력  ./data/{king_code}_{idx}.json

  - idx 앞자리가 '1'인 파일만 처리 (즉위 후 yy년 기록)
  - 앞자리가 '1'이 아닌 파일은 스킵

사용법:
    python batch_convert.py [--xml-dir ./xml] [--data-dir ./data]
"""

import json
import re
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET


# ── 왕 코드 → 왕 이름 매핑 ─────────────────────────────────────────────────
KING_CODE_MAP = {
    "waa": "태조",   "wba": "정종",   "wca": "태종",   "wda": "세종",
    "wea": "문종",   "wfa": "단종",   "wga": "세조",   "wha": "예종",
    "wia": "성종",   "wja": "연산군", "wka": "중종",   "wla": "인종",
    "wma": "명종",   "wna": "선조",   "wnb": "선조",
    "woa": "광해군", "wob": "광해군",
    "wpa": "인조",   "wqa": "효종",
    "wra": "현종",   "wrb": "현종",
    "wsa": "숙종",   "wsb": "숙종",
    "wta": "경종",   "wtb": "경종",
    "wua": "영조",   "wva": "정조",   "wwa": "순조",
    "wxa": "헌종",   "wya": "철종",   "wza": "고종",
    "wzb": "순종",   "wzc": "순종",
}

# 파일명 패턴: 2nd_{king_code}_{idx}.xml
FILENAME_RE = re.compile(r"^2nd_([a-z]{3})_(\d+)\.xml$")


def should_process(idx_str: str) -> bool:
    """idx 앞자리가 '1'인 경우만 처리 (즉위 후 재위 연도 기록)."""
    return idx_str.startswith("1")


def clean_class_name(raw: str) -> str:
    """'인물(人物)' → '인물',  '왕실-의식(儀式)' → '왕실-의식'"""
    return re.sub(r"\(.*?\)", "", raw).strip()


def get_solar_year(level4_elem) -> int | None:
    """level4 내 <dateOccured type='서기'>의 date 속성에서 연도 추출."""
    for d in level4_elem.iter("dateOccured"):
        if d.get("type") == "서기":
            m = re.match(r"(\d{4})", d.get("date", ""))
            if m:
                return int(m.group(1))
    return None


def parse_reign_date(level4_elem):
    """
    <dateOccured type='재위연도'>태조 1년 7월 17일</dateOccured> 파싱.
    윤달 예: '태조 1년 윤7월 17일'
    반환: (year_str, month_str, day_str)  ex) ("1년", "윤7월", "17일")
    """
    for d in level4_elem.iter("dateOccured"):
        if d.get("type") == "재위연도":
            text = d.text or ""
            m = re.search(r"(\d+)년\s*(윤)?(\d+)월\s*(\d+)일", text)
            if m:
                year_str  = f"{m.group(1)}년"
                month_str = f"{'윤' if m.group(2) else ''}{m.group(3)}월"
                day_str   = f"{m.group(4)}일"
                return year_str, month_str, day_str
    return "", "", ""


def convert_xml(xml_path: Path, king_code: str) -> list[dict]:
    """XML 파일 하나를 파싱해 record 리스트를 반환."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    king_name = KING_CODE_MAP.get(king_code, "")
    records   = []

    for level4 in root.iter("level4"):
        solar_year              = get_solar_year(level4)
        year_str, month_str, day_str = parse_reign_date(level4)

        for level5 in level4.iter("level5"):
            article_id = level5.get("id", "")
            biblio     = level5.find("front/biblioData")
            if biblio is None:
                continue

            # 제목
            title_elem = biblio.find("title/mainTitle")
            title = (title_elem.text or "").strip() if title_elem is not None else ""

            # 기사 순번: <docNo> 우선, 없으면 article_id 끝 숫자
            doc_no_elem = biblio.find("docNo")
            if doc_no_elem is not None and doc_no_elem.text:
                idx = int(doc_no_elem.text.strip())
            else:
                m_idx = re.search(r"_(\d+)$", article_id)
                idx   = int(m_idx.group(1)) if m_idx else 0

            # 분류
            classes = [
                clean_class_name(sc.text)
                for sc in biblio.findall("subjectClass")
                if sc.text
            ]

            # article_id 앞 3자로 왕 코드 재확인 (혼합 파일 대비)
            kc        = article_id[:3] if article_id else king_code
            king_name = KING_CODE_MAP.get(kc, king_name)

            records.append({
                "title": title,
                "metadata": {
                    "king"      : king_name,
                    "solar_year": solar_year,
                    "article_id": article_id,
                    "year"      : year_str,
                    "month"     : month_str,
                    "day"       : day_str,
                    "idx"       : idx,
                    "class"     : classes,
                },
            })

    return records


def batch_convert(xml_dir: Path, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.glob("2nd_*.xml"))
    if not xml_files:
        print(f"⚠️  {xml_dir} 에서 처리할 XML 파일을 찾지 못했습니다.")
        return

    skipped = processed = 0

    for xml_path in xml_files:
        m = FILENAME_RE.match(xml_path.name)
        if not m:
            print(f"  skip (패턴 불일치): {xml_path.name}")
            skipped += 1
            continue

        king_code, idx_str = m.group(1), m.group(2)

        if not should_process(idx_str):
            print(f"  skip (앞자리 ≠ 1): {xml_path.name}")
            skipped += 1
            continue

        out_path = data_dir / f"{king_code}_{idx_str}.json"

        try:
            records = convert_xml(xml_path, king_code)
        except ET.ParseError as e:
            print(f"  ❌ XML 파싱 오류 [{xml_path.name}]: {e}")
            skipped += 1
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"  ✅ {xml_path.name} → {out_path.name}  ({len(records)}건)")
        processed += 1

    print(f"\n완료: {processed}개 변환, {skipped}개 스킵")


def main():
    parser = argparse.ArgumentParser(description="조선왕조실록 XML → JSON 배치 변환")
    parser.add_argument("--xml-dir",  default="./xml",  help="XML 입력 폴더 (기본: ./xml)")
    parser.add_argument("--data-dir", default="./data", help="JSON 출력 폴더 (기본: ./data)")
    args = parser.parse_args()

    batch_convert(Path(args.xml_dir), Path(args.data_dir))


if __name__ == "__main__":
    main()