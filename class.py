import json
import os

def get_all_unique_classes(folder_path):
    all_unique_classes = set()
    
    # 1. 폴더 존재 여부 확인
    if not os.path.isdir(folder_path):
        return f"폴더를 찾을 수 없습니다: {folder_path}"

    # 2. 폴더 내 모든 파일 목록 가져오기
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    if not files:
        return "해당 폴더에 JSON 파일이 없습니다."

    print(f"총 {len(files)}개의 파일을 처리 중입니다...")

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    # metadata 내의 class 리스트 추출
                    classes = data.get('metadata', {}).get('class', [])
                    all_unique_classes.update(classes)
                    
        except Exception as e:
            print(f"파일 처리 중 오류 발생 ({file_name}): {e}")

    # 결과 반환 (리스트로 변환 및 정렬)
    return sorted(list(all_unique_classes))

# --- 실행 부분 ---
folder_path = 'data'  # 분석할 폴더 경로
result = get_all_unique_classes(folder_path)

if isinstance(result, list):
    print(f"\n✅ 최종 발견된 클래스 목록 ({len(result)}개):")
    for item in result:
        print(f"- {item}")
else:
    print(result)