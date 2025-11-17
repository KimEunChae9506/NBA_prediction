# -*- coding: utf-8 -*-
"""
final.json (JSONL) 각 줄을 읽어서:
  1) subject + body → 최소 정규화
  2) Mecab(은전한닢)으로 명사/동사만 추출
  3) KNU 감성사전으로 각 토큰 polarity 조회(-2,-1,0,1,2)
  4) 평균 점수를 "emoji_score" 로 저장
  5) 최종 결과를 final_score.json (JSONL)로 저장

필수 파일 구조(예시):
  ./final.json
  ./knusl.py
  ./data/SentiWord_info.json     <-- KNU 감성사전 JSON (knusl.py가 이 경로로 읽음)
"""
import json
import re
import sys
from pathlib import Path

# --- 0) KNU 감성사전 모듈 (사용법은 업로드된 knusl.py 기준) ---
from knusl import KnuSL  # :contentReference[oaicite:1]{index=1}

INPUT_PATH  = "final_data.json"         # 입력(JSON Lines)
OUTPUT_PATH = "final_score_dict.json"   # 출력(JSON Lines)

# --- 1) 간단 정규화: URL 치환, 반복문자 축약, 공백 정리 ---
URL_PAT    = re.compile(r"https?://\S+|www\.\S+")
REPEAT_PAT = re.compile(r"(ㅋ|ㅎ|ㅠ|ㅜ)\1{2,}")
SPACE_PAT  = re.compile(r"\s+")

def normalize_minimal(text: str) -> str:
    if text is None:
        return ""
    s = URL_PAT.sub(" <URL> ", text)
    s = REPEAT_PAT.sub(r"\1\1", s)  # ㅋㅋㅋㅋ -> ㅋㅋ
    s = SPACE_PAT.sub(" ", s).strip()
    return s

# --- 2) 형태소 분석: Mecab 사용(없으면 간단 토크나이저 폴백) ---
def get_tokenizer():
    try:
        from konlpy.tag import Mecab
        mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')
        def tokenize(text: str):
            # 명사/동사만 남김 (NN*, VV*)
            return [w for w, pos in mecab.pos(text) if pos.startswith("NN") or pos.startswith("VV")]
        return tokenize, "mecab"
    except Exception:
        def tokenize_fallback(text: str):
            # 아주 단순한 폴백: 한글/영문/숫자 토큰
            return re.findall(r"[가-힣A-Za-z0-9]+", text)
        return tokenize_fallback, "regex-fallback"

tokenize, tok_backend = get_tokenizer()

# --- 3) KNU 사전 조회: knusl.KnuSL.data_list(token) → (어근, polarity) ---
# knusl.py는 print를 수행하지만 여기선 반환값만 사용
def knu_polarity(token: str) -> float:
    try:
        _, pol = KnuSL.data_list(token)  # 문자열: "-2" ~ "2" 또는 "None"
        if pol is None or pol == "None" or pol == "":
            return 0.0
        return float(pol)
    except Exception:
        return 0.0

# --- 4) 한 줄 처리: emoji_score 계산(토큰별 polarity 평균) ---
def compute_emoji_score(text: str) -> float:
    # 정규화 → 토큰화 → 사전 점수 조회
    norm = normalize_minimal(text)
    tokens = tokenize(norm)
    if not tokens:
        return 0.0
    vals = []
    for t in tokens:
        p = knu_polarity(t)
        vals.append(p)
    # 단순 평균 (토큰이 많을수록 안정화됨)
    return float(sum(vals)) / len(vals) if vals else 0.0

# --- 5) 메인 루프: 입력 JSONL → 출력 JSONL ---
def main():
    in_path  = Path(INPUT_PATH)
    out_path = Path(OUTPUT_PATH)

    if not in_path.exists():
        print(f"[ERROR] 입력 파일 없음: {in_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    # knusl.py가 기대하는 사전 파일(data/SentiWord_info.json)이 같은 폴더에 있어야 함
    if not Path("SentiWord_info.json").exists():
        print("[WARN] data/SentiWord_info.json 이 보이지 않습니다. knusl.py 경로를 확인하세요.", file=sys.stderr)

    n, w = 0, 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            subject = obj.get("subject", "")
            body    = obj.get("body", "")
            text    = f"{subject}\n{body}".strip()

            score = compute_emoji_score(text)
            obj["emoji_score"] = round(score, 6)  # 요청 필드명 그대로

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    print(f"[{tok_backend}] 완료: {n} lines → {out_path}")

if __name__ == "__main__":
    main()
