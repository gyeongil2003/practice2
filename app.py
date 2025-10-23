# app.py
# Streamlit 지하철 승하차 분석/시각화 데모 (서울 열린데이터 '호선별·역별·시간대별 승하차')
# - 데이터 적재(인코딩 자동 감지), 전처리(롱포맷), 대시보드(필터, KPI, 시각화), 단순 예측(이동평균)
# - 파일 경로/URL/업로드 모두 지원
# - GitHub Raw CSV도 그대로 사용 가능

import io
import re
import sys
import time
import typing as T
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="🔮 지하철 승하차 분석 대시보드",
    page_icon="🚇",
    layout="wide"
)

# -----------------------------
# 0) 유틸: 인코딩 자동 감지 로더
# -----------------------------
def read_csv_smart(source: T.Union[str, io.BytesIO]) -> pd.DataFrame:
    """
    CSV 인코딩을 utf-8, utf-8-sig, cp949, euc-kr 순으로 시도하여 읽음.
    """
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            if isinstance(source, io.BytesIO):
                source.seek(0)
                return pd.read_csv(source, encoding=enc)
            else:
                return pd.read_csv(source, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


# --------------------------------
# 1) 데이터 로딩 & 캐싱
# --------------------------------
@st.cache_data(show_spinner=True)
def load_data(path_or_url: str = None, uploaded_file: io.BytesIO = None) -> pd.DataFrame:
    """
    - path_or_url 가 주어지면 해당 경로/URL에서 로드
    - uploaded_file 이 주어지면 업로드 파일에서 로드
    - 둘 다 None이면, 현재 디렉토리 station.csv 시도
    """
    if uploaded_file is not None:
        df = read_csv_smart(uploaded_file)
    elif path_or_url:
        df = read_csv_smart(path_or_url)
    else:
        # 로컬 동작: 앱과 같은 폴더의 station.csv 시도
        df = read_csv_smart("station.csv")

    # 기본 컬럼 체크
    required = ["사용월", "호선명", "지하철역"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼 '{c}' 이(가) 없음. 현재 컬럼: {list(df.columns)}")

    # 시간대 열만 추려 타입 확인 (정수/실수 변환)
    time_cols = [c for c in df.columns if re.search(r"\d{2}시-\d{2}시\s+(승차인원|하차인원)", c)]
    # 숫자 변환(콤마, 공백 방지)
    for c in time_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int64)

    # 타입 최적화
    df["호선명"] = df["호선명"].astype("category")
    df["지하철역"] = df["지하철역"].astype("category")

    # 사용월 -> datetime(말일 가정) 또는 period 처리
    # 데이터가 202509 같은 정수형으로 오는 경우가 많음
    def parse_yyyymm(x):
        x = str(int(x))
        year, month = int(x[:4]), int(x[4:6])
        return pd.Timestamp(year=year, month=month, day=1)

    df["사용월"] = df["사용월"].apply(parse_yyyymm)

    return df


# --------------------------------
# 2) 전처리: Wide -> Long
# --------------------------------
@st.cache_data(show_spinner=False)
def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    시간대별 승/하차 열을 Long 포맷으로 변환
    columns 예: '07시-08시 승차인원', '07시-08시 하차인원'
    -> ['시간대', '구분(승차/하차)', '인원']
    """
    time_cols = [c for c in df.columns if re.search(r"\d{2}시-\d{2}시\s+(승차인원|하차인원)", c)]
    id_vars = [c for c in df.columns if c not in time_cols]

    long_df = df.melt(id_vars=id_vars, value_vars=time_cols, var_name="항목", value_name="인원")
    # 항목 -> 시간대 / 구분
    # 예: "07시-08시 승차인원"
    # 그룹 추출
    pat = r"(?P<시간대>\d{2}시-\d{2}시)\s+(?P<구분>승차인원|하차인원)"
    extracted = long_df["항목"].str.extract(pat)
    long_df["시간대"] = extracted["시간대"]
    long_df["구분"] = extracted["구분"].str.replace("인원", "", regex=False)  # '승차', '하차'로 정리
    long_df = long_df.drop(columns=["항목"])

    # 시간대 정렬용 order 컬럼
    hours_order = [
        "04시-05시","05시-06시","06시-07시","07시-08시","08시-09시","09시-10시",
        "10시-11시","11시-12시","12시-13시","13시-14시","14시-15시","15시-16시",
        "16시-17시","17시-18시","18시-19시","19시-20시","20시-21시","21시-22시",
        "22시-23시","23시-24시","00시-01시","01시-02시","02시-03시","03시-04시"
    ]
    hour_order_map = {h:i for i,h in enumerate(hours_order)}
    long_df["시간대순서"] = long_df["시간대"].map(hour_order_map)
    long_df["시간대"] = pd.Categorical(long_df["시간대"], categories=hours_order, ordered=True)
    long_df["구분"] = pd.Categorical(long_df["구분"], categories=["승차","하차"], ordered=False)

    return long_df


# --------------------------------
# 3) 사이드바: 데이터 소스 & 필터
# --------------------------------
st.sidebar.header("⚙️ 데이터 설정")

source_mode = st.sidebar.radio(
    "데이터 소스 선택",
    options=["로컬 파일(station.csv)", "URL(예: GitHub Raw)", "직접 업로드"],
    index=0
)

path_or_url = None
uploaded = None
if source_mode == "URL(예: GitHub Raw)":
    path_or_url = st.sidebar.text_input(
        "CSV URL 입력",
        value="",
        placeholder="https://raw.githubusercontent.com/<user>/<repo>/main/station.csv"
    )
elif source_mode == "직접 업로드":
    uploaded = st.sidebar.file_uploader("CSV 업로드", type=["csv"])

with st.sidebar:
    st.caption("💡 GitHub에 올린 CSV의 Raw URL을 붙여넣으면 바로 불러옵니다.")

# 데이터 로딩
with st.spinner("데이터 불러오는 중..."):
    df = load_data(path_or_url=path_or_url if path_or_url else None, uploaded_file=uploaded)

long_df = to_long_format(df)

# 필터 UI
st.sidebar.header("🔎 필터")
months = sorted(long_df["사용월"].dt.strftime("%Y-%m").unique())
sel_months = st.sidebar.multiselect("사용월(복수 선택 가능)", options=months, default=months[-1:])
lines = sorted(long_df["호선명"].cat.categories.tolist())
sel_lines = st.sidebar.multiselect("호선", options=lines, default=lines)
# 선택된 월 문자열 -> Timestamp로 역변환
sel_months_ts = pd.to_datetime(sel_months, format="%Y-%m") if sel_months else []

# 역 이름 동적 필터
filtered_for_stations = long_df[long_df["호선명"].isin(sel_lines)]
stations = sorted(filtered_for_stations["지하철역"].cat.categories.tolist())
sel_stations = st.sidebar.multiselect("지하철역 (선택 없으면 전체)", options=stations, default=[])

view_mode = st.sidebar.radio("구분", options=["전체(승/하차 합계)", "승차", "하차"], index=0)

# 적용
mask = long_df["사용월"].isin(sel_months_ts) if len(sel_months_ts) > 0 else True
mask &= long_df["호선명"].isin(sel_lines)
if sel_stations:
    mask &= long_df["지하철역"].isin(sel_stations)

dfv = long_df[mask].copy()

# 구분 처리
if view_mode == "승차":
    dfv = dfv[dfv["구분"] == "승차"]
elif view_mode == "하차":
    dfv = dfv[dfv["구분"] == "하차"]
else:
    # 합계 컬럼으로 집계
    dfv = (
        dfv.groupby(["사용월","호선명","지하철역","시간대","시간대순서"], as_index=False)["인원"]
        .sum()
    )
    dfv["구분"] = "합계"

# --------------------------------
# 4) 상단 KPI
# --------------------------------
st.title("🚇 서울 지하철 승하차 분석 대시보드")
st.caption("데이터: 서울열린데이터광장 ‘지하철 호선별·역별 시간대별 승하차 인원’")

col1, col2, col3, col4 = st.columns(4)
total_people = int(dfv["인원"].sum())
peak_row = dfv.groupby("시간대", as_index=False)["인원"].sum().sort_values("인원", ascending=False).head(1)
peak_hour = peak_row["시간대"].iloc[0] if not peak_row.empty else "-"
peak_val = int(peak_row["인원"].iloc[0]) if not peak_row.empty else 0
num_stations = dfv["지하철역"].nunique()
num_lines = dfv["호선명"].nunique()

col1.metric("총 인원(필터 적용)", f"{total_people:,}")
col2.metric("피크 시간대", f"{str(peak_hour)}", f"{peak_val:,} 명")
col3.metric("역 개수", f"{num_stations:,}")
col4.metric("호선 수", f"{num_lines:,}")

# --------------------------------
# 5) 시각화: 시간대 추이 (선택 구분)
# --------------------------------
st.subheader("⏱️ 시간대별 추이")
st.caption("필터된 조건에서 시간대별 인원 변화를 확인합니다.")

# 집계: (월별을 묶고) 시간대별, 호선/역 선택에 따라 라인 구분
line_mode = st.radio("라인 구분", options=["전체 합", "호선별", "역별"], horizontal=True, index=0)

chart_df = dfv.copy()
# 월 다중일 때는 월 합계
chart_df = chart_df.groupby(["호선명","지하철역","시간대","시간대순서"], as_index=False)["인원"].sum()

if line_mode == "전체 합":
    plot_df = chart_df.groupby(["시간대","시간대순서"], as_index=False)["인원"].sum()
    color = alt.value("#4c78a8")
    key = "전체"
    plot_df["그룹"] = key
elif line_mode == "호선별":
    plot_df = chart_df.groupby(["호선명","시간대","시간대순서"], as_index=False)["인원"].sum()
    plot_df = plot_df.rename(columns={"호선명":"그룹"})
    color = "그룹:N"
else:
    plot_df = chart_df.groupby(["지하철역","시간대","시간대순서"], as_index=False)["인원"].sum()
    plot_df = plot_df.rename(columns={"지하철역":"그룹"})
    color = "그룹:N"

line_chart = (
    alt.Chart(plot_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("시간대:N", sort="ascending", title="시간대"),
        y=alt.Y("인원:Q", title="인원(명)"),
        color=color,
        tooltip=["그룹","시간대","인원"]
    )
    .properties(height=320)
)
st.altair_chart(line_chart, use_container_width=True)

# --------------------------------
# 6) Top-N 역 막대그래프
# --------------------------------
st.subheader("🏆 Top 역 (필터 기준 합계)")
topN = st.slider("표시할 역 개수", min_value=5, max_value=30, value=10, step=1)
top_df = (
    dfv.groupby(["지하철역"], as_index=False)["인원"].sum()
      .sort_values("인원", ascending=False)
      .head(topN)
)
bar = (
    alt.Chart(top_df)
    .mark_bar()
    .encode(
        x=alt.X("인원:Q", title="인원(명)"),
        y=alt.Y("지하철역:N", sort="-x", title=None),
        tooltip=["지하철역","인원"]
    )
    .properties(height=30 * len(top_df))
)
st.altair_chart(bar, use_container_width=True)

# --------------------------------
# 7) 호선 × 시간대 히트맵
# --------------------------------
st.subheader("🌡️ 히트맵: 호선 × 시간대")
hm_df = (
    dfv.groupby(["호선명","시간대","시간대순서"], as_index=False)["인원"].sum()
)
heat = (
    alt.Chart(hm_df)
    .mark_rect()
    .encode(
        x=alt.X("시간대:N", sort="ascending", title="시간대"),
        y=alt.Y("호선명:N", sort="ascending", title="호선"),
        color=alt.Color("인원:Q", title="인원(명)"),
        tooltip=["호선명","시간대","인원"]
    )
    .properties(height=28 * hm_df["호선명"].nunique() + 60)
)
st.altair_chart(heat, use_container_width=True)

# --------------------------------
# 8) 단순 예측(데모): 최근 N개월 이동평균으로 다음달 시간대 패턴 예측
# --------------------------------
st.subheader("🧪 예측(데모): 최근 N개월 이동평균 → 다음달 시간대별 예상")
st.caption("주의: ‘요일’ 등 변수가 없어 **학습 없이** 단순 이동평균으로 예측합니다(데모용).")

with st.expander("예측 설정 열기"):
    n_months = st.slider("최근 N개월", min_value=2, max_value=12, value=3)
    target_line = st.selectbox("호선 선택(예측)", options=lines)
    # 해당 호선의 역 목록
    stations_for_line = sorted(long_df[long_df["호선명"] == target_line]["지하철역"].cat.categories.tolist())
    target_station = st.selectbox("역 선택(예측)", options=stations_for_line)

btn = st.button("예측 실행")
if btn:
    # 특정 호선/역만 추출
    ts_df = long_df[(long_df["호선명"] == target_line) & (long_df["지하철역"] == target_station)].copy()
    # 월-시간대-구분 기준 집계
    ts_df = ts_df.groupby(["사용월","시간대","구분"], as_index=False)["인원"].sum()
    # 최근 N개월
    last_months = sorted(ts_df["사용월"].unique())[-n_months:]
    recent = ts_df[ts_df["사용월"].isin(last_months)]

    # 시간대/구분별 평균 → 다음달 예상
    pred = (
        recent.groupby(["시간대","구분"], as_index=False)["인원"].mean()
              .rename(columns={"인원":"예상인원"})
    )
    # 시각화
    pred_chart = (
        alt.Chart(pred)
        .mark_line(point=True)
        .encode(
            x=alt.X("시간대:N", sort="ascending"),
            y=alt.Y("예상인원:Q"),
            color="구분:N",
            tooltip=["시간대","구분","예상인원"]
        )
        .properties(title=f"{target_line} {target_station} – 다음달 시간대별 예상(최근 {n_months}개월 평균)")
    )
    st.altair_chart(pred_chart, use_container_width=True)

# --------------------------------
# 9) 데이터 내려받기
# --------------------------------
st.subheader("⬇️ 필터 적용 데이터 내려받기 (CSV)")
csv_bytes = dfv.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="CSV 다운로드",
    data=csv_bytes,
    file_name="filtered_subway.csv",
    mime="text/csv"
)

# --------------------------------
# 10) 데이터 미리보기
# --------------------------------
st.subheader("👀 데이터 미리보기")
st.dataframe(dfv.head(200))
