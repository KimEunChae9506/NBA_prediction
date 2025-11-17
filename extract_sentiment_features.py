# -*- coding: utf-8 -*-
"""
Stage2 완전 통합 파이프라인 (요청 반영 버전)
- (1) time_slot 기반 구간별 감성점수 필드 추가 (slot_team_mean, slot_n)
- (2) 팀별 감성격차 필드 추가 (slot_opp_mean_final, slot_delta_sent_final)
- (3) 부상여부 더미 변수 추가 (injury_yn: y/n, injury_slot: Q1~Q8 or None)
    * SportsData InjuredPlayers 응답 스키마 반영:
      - Team → 소문자
      - InjuryStartDate → ET(미국 동부시간)으로 해석
    * 게시글 regdate(KST) → ET로 변환해서 날짜(YYYYMMDD) 비교
    * 매칭 우선순위: game_key 날짜+팀 → regdate(ET) 날짜+팀 → regdate(ET) 날짜만

입력:
  - final_stage1_features.jsonl
    (필수: id, subject, body, regdate, teams_id, result|y, time_slot, game_key, sent_score_model, sent_score_lexicon)
  - game_timeslots.csv
    (필수: game_key, Q1..Q8 — 모든 시각은 Asia/Seoul 기준 "구간 시작 시각")

외부 API:
  - SportsData InjuredPlayers
    https://api.sportsdata.io/v3/nba/projections/json/InjuredPlayers?key=<API_KEY>

출력:
  - final_stage2_with_injury.jsonl
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
import re

import pandas as pd
import numpy as np
import requests

try:
    from zoneinfo import ZoneInfo
    EASTERN = ZoneInfo("America/New_York")  # ET
    SEOUL   = ZoneInfo("Asia/Seoul")        # KST
except Exception:
    # 아주 오래된 파이썬 환경 대비 (fallback, DST 미지원)
    EASTERN = timezone(timedelta(hours=-5))
    SEOUL   = timezone(timedelta(hours=9))

SLOT_COLS = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8"]


# -----------------------------
# 공통 I/O 유틸
# -----------------------------
def load_jsonl_to_df(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    if not rows:
        raise ValueError(f"No valid rows in: {path}")
    return pd.DataFrame(rows)


def write_df_to_jsonl(df: pd.DataFrame, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------------
# 시간 파싱 유틸
# -----------------------------

import numpy as np
import re

def parse_like_regdate_kst(s: str) -> datetime | None:
    """
    '20250623092400000' 같은 KST 타임스탬프(문자만) 포함 문자열을 KST aware datetime으로 파싱.
    """
    if s is None:
        return None
    ss = str(s).strip()
    digits = re.sub(r"[^\d]", "", ss)
    # 길이에 맞춰 자르기 (YYYYMMDDHHMMSS)
    if len(digits) >= 14:
        y, m, d = int(digits[0:4]), int(digits[4:6]), int(digits[6:8])
        H, M, S = int(digits[8:10]), int(digits[10:12]), int(digits[12:14])
        return datetime(y, m, d, H, M, S, tzinfo=ASIA_SEOUL)
    if len(digits) >= 8:
        y, m, d = int(digits[0:4]), int(digits[4:6]), int(digits[6:8])
        return datetime(y, m, d, 0, 0, 0, tzinfo=ASIA_SEOUL)
    return None

def kst_regdate_to_et_ymd(reg: str) -> str | None:
    dt_kst = parse_like_regdate_kst(reg)
    if not dt_kst:
        return None
    dt_et = dt_kst.astimezone(ET)
    return dt_et.strftime("%Y%m%d")

def ymd_from_game_key(game_key: str) -> str | None:
    try:
        return str(game_key).split("_")[0]
    except Exception:
        return None

def add_injury_dummy_date_team(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    날짜+팀 매칭일 때만 injury_yn='y'.
    우선순위:
      (A) game_key의 날짜 + teams_id(소문자)
      (B) regdate(ET 변환) 날짜 + teams_id(소문자)
    둘 중 하나라도 부상목록(ET 날짜+팀)에 존재하면 'y'.
    injury_slot은 항상 None.
    """
    raw = fetch_injuries(api_key=api_key)
    inj = normalize_injury_records(raw)

    if inj.empty:
        df["injury_yn"] = "n"
        df["injury_slot"] = None
        return df

    # 부상 (ymd_et, team) 세트
    inj_keyset = set(zip(inj["injury_ymd_et"].astype(str), inj["team_api"].astype(str)))

    # 게시글 측 조인키 준비
    df["_team_lower"] = df["teams_id"].astype(str).str.lower()
    df["_ymd_game"]   = df["game_key"].apply(ymd_from_game_key)
    df["_ymd_reg_et"] = df["regdate"].apply(kst_regdate_to_et_ymd)

    cond_game = list(zip(df["_ymd_game"].astype(str), df["_team_lower"]))  # (날짜, 팀)
    cond_reg  = list(zip(df["_ymd_reg_et"].astype(str), df["_team_lower"]))

    # 날짜+팀 매칭
    yn = []
    for key_game, key_reg in zip(cond_game, cond_reg):
        flag = (key_game in inj_keyset) or (key_reg in inj_keyset)
        yn.append("y" if flag else "n")

    df["injury_yn"] = yn
    df["injury_slot"] = None

    # 작업 컬럼 정리
    df.drop(columns=["_team_lower","_ymd_game","_ymd_reg_et"], errors="ignore", inplace=True)
    return df


def parse_regdate_kst(reg: Any) -> Optional[datetime]:
    """
    게시글 regdate를 'KST aware datetime'으로 파싱.
    입력은 20250623092400000 같은 숫자문자열/혼합 문자열/ISO 이하 다양성 가정.
    """
    if reg is None:
        return None
    s = str(reg).strip()
    # ISO 우선
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=SEOUL)
        else:
            dt = dt.astimezone(SEOUL)
        return dt
    except Exception:
        pass

    # 숫자만 추출 후 길이별 파싱
    just = re.sub(r"[^\d]", "", s)
    fmts = ["%Y%m%d%H%M%S%f","%Y%m%d%H%M%S","%Y%m%d%H%M","%Y%m%d%H","%Y%m%d"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(just[:len(datetime.now().strftime("%Y%m%d%H%M%S"))], fmt)
            return dt.replace(tzinfo=SEOUL)
        except Exception:
            continue

    # 마지막 시도: 'YYYY-MM-DD HH:MM:SS'
    try:
        dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=SEOUL)
        return dt
    except Exception:
        return None


def kst_to_et_date_str(reg: Any) -> Optional[str]:
    """
    KST regdate → ET로 변환 → 'YYYYMMDD' 문자열 반환
    """
    dt_kst = parse_regdate_kst(reg)
    if dt_kst is None:
        return None
    return dt_kst.astimezone(EASTERN).strftime("%Y%m%d")


from datetime import datetime, timezone, timedelta
import pandas as pd

ET = timezone(timedelta(hours=-5))   # SportsData는 미국 동부(ET) 기준 날짜로 간주
ASIA_SEOUL = timezone(timedelta(hours=9))

def parse_sportsdata_injury_dt_et(s: str) -> datetime | None:
    if not s:
        return None
    try:
        # 예: "2025-05-12T00:00:00"
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            # 명시 없는 경우 ET로 해석
            dt = dt.replace(tzinfo=ET)
        else:
            # 타 TZ면 ET로 변환
            dt = dt.astimezone(ET)
        return dt
    except Exception:
        return None

def normalize_injury_records(raw: list[dict]) -> pd.DataFrame:
    """
    반환 컬럼:
      team_api (소문자 3자 팀코드), injury_dt_et (aware), injury_ymd_et ('YYYYMMDD')
    """
    recs = []
    for r in raw:
        team = r.get("Team")
        dt_et = parse_sportsdata_injury_dt_et(r.get("InjuryStartDate"))
        recs.append({
            "team_api": (str(team).lower() if team else None),
            "injury_dt_et": dt_et,
            "injury_ymd_et": (dt_et.strftime("%Y%m%d") if dt_et else None),
        })
    df = pd.DataFrame(recs)
    if not df.empty:
        df = df[~df["injury_dt_et"].isna() & df["team_api"].notna()].copy()
    return df


def ymd_from_game_key(game_key: Any) -> Optional[str]:
    try:
        return str(game_key).split("_")[0]
    except Exception:
        return None


# -----------------------------
# 슬롯/경기 타임라인
# -----------------------------
def load_timeslot_csv(path: str) -> Dict[str, Dict[str, datetime]]:
    """
    game_timeslots.csv 를 읽어 {game_key: {Q1..Q8: datetime(KST tz-aware)}} 매핑 생성
    """
    df = pd.read_csv(path)
    for c in SLOT_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    mapping: Dict[str, Dict[str, datetime]] = {}
    for _, row in df.iterrows():
        gk = str(row["game_key"])
        slots = {}
        ok = True
        for c in SLOT_COLS:
            t = row[c]
            if pd.isna(t):
                ok = False
                break
            # pandas Timestamp → tz 보정(KST)
            if getattr(t, "tzinfo", None) is None:
                t = t.tz_localize(SEOUL)
            else:
                t = t.tz_convert(SEOUL)
            slots[c] = t.to_pydatetime()
        if ok:
            mapping[gk] = slots
    return mapping


def assign_slot_from_qstarts_kst(dt_et: datetime, qstarts_kst: Dict[str, datetime]) -> Optional[str]:
    """
    부상 시각(ET)을 KST로 변환해 슬롯 경계(KST)와 비교해 Q1~Q8 결정.
    Q1 이전은 None, Q8 이후는 Q8로 클램프.
    """
    if dt_et is None:
        return None
    dt_kst = dt_et.astimezone(SEOUL)
    if dt_kst < qstarts_kst["Q1"]:
        return None
    for i in range(1, 8):
        cur, nxt = f"Q{i}", f"Q{i+1}"
        if qstarts_kst[cur] <= dt_kst < qstarts_kst[nxt]:
            return cur
    return "Q8"


# -----------------------------
# (1) 구간별 감성점수
# -----------------------------
def add_timeslot_scores(
    df: pd.DataFrame,
    alpha: float = 0.6,
    min_posts_per_cell: int = 1,
    model_col: str = "sent_score_model",
    lex_col: str  = "sent_score_lexicon",
    game_col: str = "game_key",
    slot_col: str = "time_slot",
    team_col: str = "teams_id",
):
    # 결합 점수 생성
    for c in (model_col, lex_col):
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["combined_score"] = alpha * df[model_col] + (1.0 - alpha) * df[lex_col]

    # (game, slot, team) 평균/표본수
    grp_cols = [game_col, slot_col, team_col]
    agg = (
        df.groupby(grp_cols)["combined_score"]
          .agg(["count", "mean"])
          .reset_index()
          .rename(columns={"count": "slot_n", "mean": "slot_team_mean"})
    )
    if min_posts_per_cell > 1:
        agg.loc[agg["slot_n"] < min_posts_per_cell, "slot_team_mean"] = np.nan

    df = df.merge(agg, on=grp_cols, how="left")
    return df


# -----------------------------
# (2) 팀별 감성 격차
# -----------------------------
def add_team_gap(
    df: pd.DataFrame,
    game_col: str = "game_key",
    slot_col: str = "time_slot",
    team_col: str = "teams_id",
    team_mean_col: str = "slot_team_mean"
):
    """
    최종 추가:
      - slot_opp_mean_final
      - slot_delta_sent_final = slot_team_mean - slot_opp_mean_final
      - slot_gap_fallback_level: 0(슬롯 직비교) / 1(경기 전체 fallback) / 2(0격차)
    """
    # 슬롯 단위 wide
    wide_slot = df.pivot_table(
        index=[game_col, slot_col],
        columns=team_col,
        values=team_mean_col,
        aggfunc="mean"
    )

    slot_rows = []
    for (g, s), row in wide_slot.iterrows():
        valid = {t: row[t] for t in row.index if pd.notna(row[t])}
        if not valid:
            continue
        for my_team, _ in valid.items():
            opp_vals = [v for t, v in valid.items() if t != my_team and pd.notna(v)]
            opp_slot_mean = float(np.mean(opp_vals)) if opp_vals else np.nan
            slot_rows.append({
                game_col: g,
                slot_col: s,
                team_col: my_team,
                "slot_opp_mean_slot": opp_slot_mean
            })
    slot_level_opp = pd.DataFrame(slot_rows)

    # 경기 전체 단위 wide
    wide_game = df.pivot_table(
        index=[game_col],
        columns=team_col,
        values=team_mean_col,
        aggfunc="mean"
    )
    game_rows = []
    for g, row in wide_game.iterrows():
        valid = {t: row[t] for t in row.index if pd.notna(row[t])}
        if not valid:
            continue
        for my_team, _ in valid.items():
            opp_vals = [v for t, v in valid.items() if t != my_team and pd.notna(v)]
            opp_game_mean = float(np.mean(opp_vals)) if opp_vals else np.nan
            game_rows.append({
                game_col: g,
                team_col: my_team,
                "slot_opp_mean_game": opp_game_mean
            })
    game_level_opp = pd.DataFrame(game_rows)

    # 머지 + fallback
    df = df.merge(slot_level_opp, on=[game_col, slot_col, team_col], how="left")
    df = df.merge(game_level_opp, on=[game_col, team_col],              how="left")

    opp_final_list, fallback_level_list = [], []
    for _, r in df.iterrows():
        opp_slot = r.get("slot_opp_mean_slot", np.nan)
        opp_game = r.get("slot_opp_mean_game", np.nan)
        my_val   = r.get(team_mean_col, np.nan)
        if pd.notna(opp_slot):
            opp_final = opp_slot; level = 0
        elif pd.notna(opp_game):
            opp_final = opp_game; level = 1
        else:
            opp_final = my_val;   level = 2
        opp_final_list.append(opp_final)
        fallback_level_list.append(level)

    df["slot_opp_mean_final"]   = opp_final_list
    df["slot_gap_fallback_level"] = fallback_level_list
    df["slot_delta_sent_final"] = df[team_mean_col] - df["slot_opp_mean_final"]
    return df


# -----------------------------
# (3) 부상 여부/슬롯 라벨링
# -----------------------------
def fetch_injuries(api_key: str, timeout: int = 20) -> List[dict]:
    url = f"https://api.sportsdata.io/v3/nba/projections/json/InjuredPlayers?key={api_key}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


def add_injury_dummy(
    df: pd.DataFrame,
    timeslot_csv: str,
    api_key: str
) -> pd.DataFrame:
    # 슬롯 타임라인(KST)
    gmap = load_timeslot_csv(timeslot_csv)

    # SportsData → 정규화
    inj_raw = fetch_injuries(api_key=api_key)
    inj = normalize_injury_records(inj_raw)

    if inj.empty:
        df["injury_yn"] = "n"
        df["injury_slot"] = None
        return df

    # 조인용 보조키: 팀소문자, game_key에서 날짜, regdate(KST→ET) 날짜
    df["_team_lower"] = df["teams_id"].astype(str).str.lower()
    df["_ymd_game"]   = df["game_key"].apply(ymd_from_game_key)
    df["_ymd_reg_et"] = df["regdate"].apply(kst_to_et_date_str)

    # 인덱스 사전 구성
    # 1) (inj_ymd, team) → injury rows
    inj_by_ymd_team: Dict[tuple, List[pd.Series]] = {}
    # 2) (inj_ymd) → injury rows (팀 무시)
    inj_by_ymd: Dict[str, List[pd.Series]] = {}

    for _, r in inj.iterrows():
        ymd = r.get("injury_ymd_et")
        tlo = r.get("team_api")
        if ymd:
            inj_by_ymd.setdefault(ymd, []).append(r)
            if tlo:
                inj_by_ymd_team.setdefault((ymd, tlo), []).append(r)

    injury_yn, injury_slot = [], []

    for _, row in df.iterrows():
        gk = row.get("game_key")
        team_lo = row.get("_team_lower")
        ymd_game = row.get("_ymd_game")
        ymd_reg  = row.get("_ymd_reg_et")

        # 슬롯 경계 준비
        qstarts = gmap.get(gk)  # KST 경계
        # 매칭 후보들(우선순위대로)
        candidates: List[pd.Series] = []

        # 1) game_key 날짜 + 팀
        if ymd_game and team_lo:
            candidates.extend(inj_by_ymd_team.get((ymd_game, team_lo), []))

        # 2) regdate(ET) 날짜 + 팀
        if not candidates and ymd_reg and team_lo:
            candidates.extend(inj_by_ymd_team.get((ymd_reg, team_lo), []))

        # 3) regdate(ET) 날짜만
        if not candidates and ymd_reg:
            candidates.extend(inj_by_ymd.get(ymd_reg, []))

        if not candidates:
            injury_yn.append("n")
            injury_slot.append(None)
            continue

        # 슬롯 판정 (가능하면 injury_dt로 슬롯 매핑)
        s_found = None
        if qstarts:
            for rec in candidates:
                dt_et = rec.get("injury_dt_et")
                s = assign_slot_from_qstarts_kst(dt_et, qstarts)
                if s is not None:
                    s_found = s
                    break

        injury_yn.append("y")
        injury_slot.append(s_found)

    df["injury_yn"]   = injury_yn
    df["injury_slot"] = injury_slot

    # 보조키 제거
    df = df.drop(columns=["_team_lower","_ymd_game","_ymd_reg_et"], errors="ignore")
    return df

def add_injury_dummy_date_only(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    날짜만 맞으면 injury_yn='y' 로 지정.
    - 게임 날짜(= game_key의 YYYYMMDD) 또는
    - 게시글 regdate(KST → ET 변환)의 날짜(YYYYMMDD)
      중 하나라도 부상목록의 날짜와 같으면 y.
    - injury_slot 은 항상 None.
    """
    raw = fetch_injuries(api_key=api_key)
    inj = normalize_injury_records(raw)

    # 부상 날짜 집합(ET 기준)
    inj_days = set(inj["injury_ymd_et"].dropna().astype(str).tolist()) if not inj.empty else set()
    print(inj_days)

    if not inj_days:
        df["injury_yn"] = "n"
        df["injury_slot"] = None
        return df

    # game_key 날짜와 regdate(ET) 날짜를 뽑음
    def _ymd_from_game_key(gk: Any) -> Optional[str]:
        try:
            return str(gk).split("_")[0]
        except Exception:
            return None

    def _ymd_from_regdate_et(reg: Any) -> Optional[str]:
        return kst_to_et_date_str(reg)  # KST → ET → 'YYYYMMDD'

    ymd_game = df["game_key"].apply(_ymd_from_game_key)
    ymd_reg  = df["regdate"].apply(_ymd_from_regdate_et)

    # 날짜 매칭
    df["injury_yn"] = np.where(
        ymd_game.isin(inj_days) | ymd_reg.isin(inj_days),
        "y", "n"
    )
    df["injury_slot"] = None
    return df

# -----------------------------
# 파이프라인 실행
# -----------------------------
def run_stage2_full(
    in_jsonl: str = "final_stage1_features.jsonl",
    timeslot_csv: str = "game_timeslots.csv",
    api_key: str = "",
    out_jsonl: str = "final_stage2_with_injury.jsonl",
    alpha: float = 0.6,
    min_posts_per_cell: int = 1
):
    df = load_jsonl_to_df(in_jsonl)

    # (1) 슬롯별 감성 평균 계산
    df = add_timeslot_scores(
        df,
        alpha=alpha,
        min_posts_per_cell=min_posts_per_cell,
        model_col="sent_score_model",
        lex_col="sent_score_lexicon",
        game_col="game_key",
        slot_col="time_slot",
        team_col="teams_id",
    )

    # (2) 팀 간 감성 격차 계산
    df = add_team_gap(
        df,
        game_col="game_key",
        slot_col="time_slot",
        team_col="teams_id",
    )

    # (3) 부상 여부/슬롯 라벨링 (API)
    if api_key:
        df = add_injury_dummy_date_team(df, api_key=api_key)
    else:
        # API 키가 없으면 기본값
        df["injury_yn"] = "n"
        df["injury_slot"] = None

    # 중간 컬럼 정리
    drop_cols = [
        "combined_score",
        "slot_opp_mean_slot",
        "slot_opp_mean_game",
        "slot_opp_mean_final",
        "slot_gap_fallback_level",
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # 최종 컬럼(존재하는 것만)
    keep_cols = [
        "id","subject","body","regdate","game_key","teams_id",
        "result","y","time_slot",
        "sent_score_model","sent_score_lexicon",
        "slot_n","slot_team_mean","slot_delta_sent_final",
        "injury_yn","injury_slot",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    write_df_to_jsonl(df, out_jsonl)
    print(f"[DONE] Stage2 full → {out_jsonl}")
    print("  columns:", keep_cols)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl",     type=str, default="final_stage1_features.json")
    ap.add_argument("--timeslot_csv", type=str, default="game_timeslots.csv")
    ap.add_argument("--api_key",      type=str, default="86b0e82269e8434bbc8918e89b7e815b", help="SportsData API key (미입력시 injury_yn='n')")
    ap.add_argument("--out_jsonl",    type=str, default="final_stage2_with_injury2.json")
    ap.add_argument("--alpha",        type=float, default=0.6, help="결합 가중치 α (model α + lex (1-α))")
    ap.add_argument("--min_posts_per_cell", type=int, default=1, help="(game×slot×team) 최소 표본 수")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stage2_full(
        in_jsonl=args.in_jsonl,
        timeslot_csv=args.timeslot_csv,
        api_key=args.api_key,
        out_jsonl=args.out_jsonl,
        alpha=args.alpha,
        min_posts_per_cell=args.min_posts_per_cell
    )