# -*- coding: utf-8 -*-
"""
(1) 독립변수 생성 통합 파이프라인
-1. ELECTRA 감성점수 생성 (extract_model_4.py 호출 또는 임포트)
-2. 감성사전 감성점수 생성 (extract_emojidic_5.py 호출 또는 임포트)
-3. time slot 부여 (game_timeslots.csv의 Q1~Q8 시작시각 기반)
-4. 최종 병합: id 기준으로 sent_score_model, emoji_score, time_slot, game_key 추가

사용법
------
(기본값으로 실행)
    python build_stage1_features.py
(경로/옵션 바꾸고 싶다면 상단 CONFIG 수정 또는 CLI 인자 사용)

필수 입력
--------
- base_jsonl           : final.json (id, subject, body, regdate, teams_id 포함)
- game_timeslots_csv   : game_timeslots.csv (game_key,Q1~Q8 시작시각; Asia/Seoul)
- extract_model_4.py    : ELECTRA 점수 생성 스크립트
- extract_emojidic_5.py: 사전 점수 생성 스크립트

중간 산출(기본 경로)
-------------------
- model_out_jsonl   : final_with_model_sentiment.jsonl (id별 model 점수)
- lex_out_jsonl    : final_score_dict.json           (id별 emoji_score)
- timeslot_out_jsonl: final_with_timeslot.jsonl      (id별 time_slot, game_key)

최종 산출
---------
- final_stage1_jsonl: final_stage1_features.jsonl    (위 3개 모두 합쳐서 저장)
"""

import os, re, json, argparse, subprocess, sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# =========================
# CONFIG (필요시 수정)
# =========================
CONFIG = {
    # 입력/출력 경로
    "base_jsonl": "final_data.json",
    "game_timeslots_csv": "game_timeslots.csv",
    "model_out_jsonl": "final_score_model_sentiment.json",
    "lex_out_jsonl": "final_score_dict.json",
    "timeslot_out_jsonl": "final_with_timeslot.json",
    "final_stage1_jsonl": "final_stage1_features.json",

    # 외부 스크립트 파일명
    "model_script": "extract_model_4.py",
    "lex_script": "extract_emojidic_5.py",

    # 파이프라인 토글
    "RUN_MODEL": True,
    "RUN_LEX" : True,
    "RUN_TS"  : True,   # time slot
    "MERGE"   : True,   # 최종 병합

    # time slot 설정
    "CLAMP_AFTER_Q8": True,  # Q8 이후 글을 Q8로 라벨링(아니면 드롭)
}

ASIA_SEOUL = timezone(timedelta(hours=9))
SLOT_COLS  = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]


# =========================
# 유틸: 공통 로딩/저장
# =========================
def load_jsonl_to_list(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
            rows.append(o)
    return rows

def load_jsonl_to_map(path: str) -> Dict[str, dict]:
    store = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
            _id = o.get("id") or o.get("_id")
            if _id is None:
                continue
            store[str(_id)] = o
    return store


# =========================
# 1) ELECTRA/사전 실행 (있는 스크립트 호출)
# =========================
import os, subprocess, sys

def run_external_script(python_file: str) -> None:
    if not Path(python_file).exists():
        raise FileNotFoundError(f"Not found: {python_file}")
    print(f"[RUN] {python_file}")

    # 강제 UTF-8 모드 + 표준 입출력 UTF-8
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    completed = subprocess.run(
        [sys.executable, "-X", "utf8", python_file],
        env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise RuntimeError(f"Script failed: {python_file}")
    else:
        if completed.stdout.strip():
            print(completed.stdout.strip())



# =========================
# 2) time slot 태깅
# =========================
def parse_regdate(s: str) -> Optional[datetime]:
    """ '20250505090400000' 같은 문자열을 Asia/Seoul aware datetime으로 """
    s = re.sub(r"[^\d]", "", str(s))
    fmts = ["%Y%m%d%H%M%S%f", "%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d%H"]
    for fmt in fmts:
        try:
            need = len(datetime.now().strftime("%Y%m%d%H%M%S"))
            dt = datetime.strptime(s[:need], fmt)
            return dt.replace(tzinfo=ASIA_SEOUL)
        except Exception:
            continue
    try:
        dt = datetime.strptime(s[:8], "%Y%m%d")
        return dt.replace(tzinfo=ASIA_SEOUL)
    except Exception:
        return None

def yyyymmdd_from_regdate(regdate: str) -> str:
    return str(regdate)[:8]

def load_timeslot_csv(path: str) -> Dict[str, Dict[str, datetime]]:
    """
    CSV → {game_key: {"Q1":dt, ..., "Q8":dt}}
    모든 시각은 Asia/Seoul aware 로 변환
    """
    df = pd.read_csv(path)
    for c in SLOT_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    mapping = {}
    for _, row in df.iterrows():
        gk = str(row["game_key"])
        slots = {}
        ok = True
        for c in SLOT_COLS:
            t = row[c]
            if pd.isna(t):
                ok = False
                break
            if t.tzinfo is None:
                try:
                    t = t.tz_localize("Asia/Seoul").to_pydatetime()
                except Exception:
                    t = t.to_pydatetime().replace(tzinfo=ASIA_SEOUL)
            slots[c] = t
        if ok:
            mapping[gk] = slots
    return mapping

def find_game_key_for_post(ymd: str, team_id: str, csv_keys: List[str]) -> Optional[str]:
    """
    날짜=ymd로 시작하고, team_id가 포함된 game_key (YYYYMMDD_teamA_teamB)를 찾는다.
    여러 개 매치되면 첫 번째 반환(보통 한 날짜에 한 팀은 1경기 가정)
    """
    num = 0
    team_id = (team_id or "").lower()
    for k in csv_keys:
        if not isinstance(k, str):
            continue
        if not k.startswith(ymd + "_"):
            #print(ymd)
            #print(k)
            num +=1
            continue
        parts = k.split("_")
        if len(parts) < 3:
            continue
        tA, tB = parts[1].lower(), parts[2].lower()
        if team_id in (tA, tB):
            return k
    return None

def assign_slot_from_qstarts(post_dt: datetime, qstarts: Dict[str, datetime], clamp_after_q8: bool=True) -> Optional[str]:
    """
    - post_dt < Q1          → "pre"
    - Qk ≤ post_dt < Q(k+1) → "Qk"
    - post_dt ≥ Q8          → "Q8"(clamp) 또는 None
    """
    if post_dt < qstarts["Q1"]:
        return "Q1"
    for i in range(2, 8):
        cur, nxt = f"Q{i}", f"Q{i+1}"
        if qstarts[cur] <= post_dt < qstarts[nxt]:
            return cur
    return "Q8" if clamp_after_q8 else None

def add_time_slot(base_rows: List[dict], timeslot_map: Dict[str, Dict[str, datetime]], clamp_after_q8=True) -> List[dict]:
    csv_keys = list(timeslot_map.keys())
    out = []
    for obj in base_rows:
        regdate = obj.get("regdate")
        team_id = obj.get("teams_id")
        if regdate is None or team_id is None:
            print("regdate")
            continue
        post_dt = parse_regdate(regdate)
        if post_dt is None:
            print("post_dt")
            continue
        ymd = yyyymmdd_from_regdate(regdate)
        gk = find_game_key_for_post(ymd, team_id, csv_keys)
        if gk is None:
            print( obj.get("id"))
            #print(gk)
            continue
        slot = assign_slot_from_qstarts(post_dt, timeslot_map[gk], clamp_after_q8)
        if slot is None:
            print("slot")
            continue
        obj2 = dict(obj)
        obj2["time_slot"] = slot
        obj2["game_key"]  = gk
        out.append(obj2)
    return out


# =========================
# 3) 점수 추출 도우미
# =========================
def extract_model_score(obj: Dict[str, Any]) -> Optional[float]:
    """
    ELECTRA 점수 추출:
      1) obj['model_sentiment']['sentiment_score']
      2) obj['model_score']가 list([neg, pos])면 pos-neg
      3) obj['model_score']가 스칼라면 그대로
    """
    if isinstance(obj.get("model_sentiment"), dict):
        bs = obj["model_sentiment"].get("sentiment_score")
        if bs is not None:
            try:
                return float(bs)
            except Exception:
                pass
    if "model_score" in obj:
        v = obj["model_score"]
        try:
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                neg, pos = float(v[0]), float(v[1])
                return float(pos - neg)
            return float(v)
        except Exception:
            return None
    return None

def extract_lex_score(obj: Dict[str, Any]) -> Optional[float]:
    v = obj.get("emoji_score")
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


# =========================
# 파이프라인 러너
# =========================
def run_pipeline(
    base_jsonl: str,
    game_timeslots_csv: str,
    model_script: str,
    lex_script: str,
    model_out_jsonl: str,
    lex_out_jsonl: str,
    timeslot_out_jsonl: str,
    final_stage1_jsonl: str,
    run_model: bool = True,
    run_lex: bool = True,
    run_ts: bool = True,
    do_merge: bool = True,
    clamp_after_q8: bool = True
):
    base_path = Path(base_jsonl)
    if not base_path.exists():
        raise FileNotFoundError(f"base_jsonl not found: {base_jsonl}")

    # 1) model 실행
    #if run_model:
    #    run_external_script(model_script)
    #    if not Path(model_out_jsonl).exists():
    #        raise FileNotFoundError(f"[model] expected output not found: {model_out_jsonl}")

    # 2) 사전 실행
    #if run_lex:
    #    run_external_script(lex_script)
    #    if not Path(lex_out_jsonl).exists():
    #        raise FileNotFoundError(f"[LEXICON] expected output not found: {lex_out_jsonl}")

    # 3) time slot 부여
    if run_ts:
        ts_map = load_timeslot_csv(game_timeslots_csv)
        base_rows = load_jsonl_to_list(base_jsonl)
        posts_with_slot = add_time_slot(base_rows, ts_map, clamp_after_q8=clamp_after_q8)
        if not posts_with_slot:
            raise ValueError("No posts were assigned a time_slot. Check teams_id, regdate, and CSV.")
        # 중간 저장(옵션)
        with open(timeslot_out_jsonl, "w", encoding="utf-8") as f:
            for o in posts_with_slot:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"[TIME-SLOT] wrote: {timeslot_out_jsonl}")
    else:
        posts_with_slot = load_jsonl_to_list(timeslot_out_jsonl)

    # 4) 최종 병합
    if do_merge:
        model_map = load_jsonl_to_map(model_out_jsonl) if Path(model_out_jsonl).exists() else {}
        lex_map = load_jsonl_to_map(lex_out_jsonl) if Path(lex_out_jsonl).exists() else {}

        keep_fields = [
            "id", "subject", "body", "regdate", "teams_id",
            "result", "time_slot", "game_key",
            "sent_score_model", "sent_score_lexicon"
        ]

        with open(final_stage1_jsonl, "w", encoding="utf-8") as fout:
            cnt = 0
            for obj in posts_with_slot:
                rid = str(obj.get("id") or obj.get("_id"))

                sent_score_model = 0.0
                emoji_score = 0.0

                # model 점수
                if rid in model_map:
                    bs = extract_model_score(model_map[rid])
                    if bs is not None:
                        sent_score_model = float(bs)

                # 사전 점수
                if rid in lex_map:
                    ls = extract_lex_score(lex_map[rid])
                    if ls is not None:
                        emoji_score = float(ls)

                obj["sent_score_model"] = sent_score_model
                obj["sent_score_lexicon"] = emoji_score

                # ✅ 지정된 필드만 남기기
                filtered = {k: obj.get(k, None) for k in keep_fields}

                fout.write(json.dumps(filtered, ensure_ascii=False) + "\n")
                cnt += 1

        print(f"[FINAL] wrote filtered: {final_stage1_jsonl}")
        print(f"[FIELDS] {', '.join(keep_fields)}")


# =========================
# CLI & main
# =========================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_jsonl", default=CONFIG["base_jsonl"])
    ap.add_argument("--game_timeslots_csv", default=CONFIG["game_timeslots_csv"])
    ap.add_argument("--model_script", default=CONFIG["model_script"])
    ap.add_argument("--lex_script",  default=CONFIG["lex_script"])
    ap.add_argument("--model_out_jsonl", default=CONFIG["model_out_jsonl"])
    ap.add_argument("--lex_out_jsonl",  default=CONFIG["lex_out_jsonl"])
    ap.add_argument("--timeslot_out_jsonl", default=CONFIG["timeslot_out_jsonl"])
    ap.add_argument("--final_stage1_jsonl", default=CONFIG["final_stage1_jsonl"])
    ap.add_argument("--no_run_model", action="store_true")
    ap.add_argument("--no_run_lex",  action="store_true")
    ap.add_argument("--no_run_ts",   action="store_true")
    ap.add_argument("--no_merge",    action="store_true")
    ap.add_argument("--no_clamp_after_q8", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        base_jsonl=args.base_jsonl,
        game_timeslots_csv=args.game_timeslots_csv,
        model_script=args.model_script,
        lex_script=args.lex_script,
        model_out_jsonl=args.model_out_jsonl,
        lex_out_jsonl=args.lex_out_jsonl,
        timeslot_out_jsonl=args.timeslot_out_jsonl,
        final_stage1_jsonl=args.final_stage1_jsonl,
        run_model=not args.no_run_model if CONFIG["RUN_MODEL"] else False,
        run_lex=not args.no_run_lex if CONFIG["RUN_LEX"] else False,
        run_ts=not args.no_run_ts if CONFIG["RUN_TS"] else False,
        do_merge=not args.no_merge if CONFIG["MERGE"] else False,
        clamp_after_q8=CONFIG["CLAMP_AFTER_Q8"] and (not args.no_clamp_after_q8),
    )
