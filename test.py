import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. .env 파일에 저장된 환경 변수(API 키) 불러오기
load_dotenv()

# 2. OpenAI 클라이언트 초기화 (자동으로 환경 변수의 OPENAI_API_KEY를 참조합니다)
client = OpenAI()

def ask_gpt(prompt):
    try:
        # 3. Chat Completions API 호출 (기본적인 gpt-4o-mini 모델 사용)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 친절하고 위트 있는 AI 조수입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        # 4. 생성된 답변 텍스트만 추출해서 반환
        return response.choices[0].message.content

    except Exception as e:
        return f"에러가 발생했습니다: {e}"

# 실행 테스트
if __name__ == "__main__":
    user_question = "조선왕조실록에 대해 한 문장으로 흥미롭게 설명해 줘!"
    print(f"질문: {user_question}\n")
    
    print("답변 생성 중...")
    answer = ask_gpt(user_question)
    print(f"GPT 답변:\n{answer}")