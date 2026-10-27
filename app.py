
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="Fire Insight 2.0",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
        BASE_DIR / "data" / "processed"
        / "fire_industrial_clean.parquet"
)

STAT_PATH = BASE_DIR / "reports" / "statistical_test_results.csv"
QUALITY_PATH = BASE_DIR / "reports" / "data_quality.csv"

BLUE = "#2563EB"
GRAY = "#94A3B8"

BUILDING_COL = "건물구조_표준화"
DAMAGE_COL = "재산피해(천원)"


# ==========================================
# 2. 공통 함수
# ==========================================

@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


@st.cache_data
def load_report(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def format_money(value):
    if pd.isna(value):
        return "자료 없음"

    won = int(round(value * 1000))

    if abs(won) >= 100_000_000:
        return f"{won / 100_000_000:,.2f}억 원"

    if abs(won) >= 10_000:
        return f"{won / 10_000:,.0f}만 원"

    return f"{won:,}원"


def show_chart(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(size=13),
        margin=dict(l=20, r=20, t=60, b=30)
    )

    st.plotly_chart(fig, width="stretch")


def reset_filters():
    st.session_state["cause_filter"] = []
    st.session_state["building_filter"] = []


# ==========================================
# 3. 데이터 불러오기
# ==========================================

if not DATA_PATH.exists():
    st.error("분석용 Parquet 파일이 없습니다.")
    st.stop()

df = load_data()

required_columns = [
    "조사서번호",
    "발화요인",
    "건물구조조",
    BUILDING_COL,
    DAMAGE_COL,
    "인명피해(명)",
    "사망",
    "부상",
    "연소확대범위"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"필수 컬럼이 없습니다: {missing_columns}")
    st.info(
        "03_clean_data.py를 다시 실행하여 "
        "전처리 데이터를 갱신해 주세요."
    )
    st.stop()


# ==========================================
# 4. 사이드바 필터
# ==========================================

st.sidebar.title("🔥 Fire Insight 2.0")
st.sidebar.caption("산업시설 화재 분석 플랫폼")
st.sidebar.divider()

st.sidebar.subheader("데이터 필터")

causes = sorted(
    df["발화요인"].dropna().unique().tolist()
)

buildings = sorted(
    df[BUILDING_COL].dropna().unique().tolist()
)

selected_causes = st.sidebar.multiselect(
    "발화요인",
    options=causes,
    default=[],
    placeholder="전체 발화요인",
    key="cause_filter"
)

selected_buildings = st.sidebar.multiselect(
    "건물구조 (표준화)",
    options=buildings,
    default=[],
    placeholder="전체 건물구조",
    key="building_filter"
)

st.sidebar.button(
    "🔄 필터 초기화",
    on_click=reset_filters,
    width="stretch"
)

filtered_df = df.copy()

if selected_causes:
    filtered_df = filtered_df[
        filtered_df["발화요인"].isin(selected_causes)
    ]

if selected_buildings:
    filtered_df = filtered_df[
        filtered_df[BUILDING_COL].isin(selected_buildings)
    ]

st.sidebar.divider()

st.sidebar.metric(
    "현재 조회 데이터",
    f"{len(filtered_df):,}건"
)

st.sidebar.caption(
    "선택하지 않은 필터는 전체 범주를 포함합니다. "
    "두 필터는 AND 조건으로 적용됩니다."
)


# ==========================================
# 5. 메인 화면
# ==========================================

st.title("🔥 Fire Insight 2.0")

st.subheader(
    "산업시설 화재 발생 특성 및 피해 심각도 분석"
)

st.caption(
    "데이터 출처: 소방청 화재 데이터 | "
    "분석 대상: 산업시설 화재"
)


# ==========================================
# 6. 피해액 통계 계산
# ==========================================

all_damage = pd.to_numeric(
    df[DAMAGE_COL], errors="coerce"
)

all_damage = all_damage[
    all_damage.notna() & (all_damage >= 0)
    ]

threshold_90 = all_damage.quantile(0.90)
threshold_99 = all_damage.quantile(0.99)

damage = pd.to_numeric(
    filtered_df[DAMAGE_COL], errors="coerce"
)

valid_damage = damage[
    damage.notna() & (damage >= 0)
    ]

total_damage = valid_damage.sum()

top_damage = valid_damage[
    valid_damage >= threshold_99
    ].sum()

top_share = (
    top_damage / total_damage * 100
    if total_damage > 0 else 0.0
)


# ==========================================
# 7. 핵심 KPI
# ==========================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "화재 발생 건수",
    f"{len(filtered_df):,}건"
)

col2.metric(
    "평균 재산피해",
    format_money(valid_damage.mean())
)

col3.metric(
    "중앙값 재산피해",
    format_money(valid_damage.median())
)

col4.metric(
    "전체 99백분위 이상 피해 점유율",
    f"{top_share:.2f}%" if not filtered_df.empty else "—"
)

st.caption(
    "피해 점유율은 전체 산업시설 데이터의 "
    "99백분위 기준을 현재 필터 결과에 적용합니다."
)

if filtered_df.empty:
    st.warning(
        "현재 조건에 해당하는 데이터가 없습니다. "
        "사이드바의 필터 초기화 버튼을 눌러주세요."
    )

st.divider()


# ==========================================
# 8. 분석 탭
# ==========================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 종합 현황",
    "💰 재산피해",
    "🚑 인명피해",
    "📈 통계 검정",
    "🗂️ 데이터 품질"
])


# ==========================================
# TAB 1. 종합 현황
# ==========================================

with tab1:

    st.header("01. 발화요인별 화재 발생 현황")

    if filtered_df.empty:
        st.info("표시할 데이터가 없습니다.")

    else:
        cause_count = (
            filtered_df["발화요인"]
            .value_counts()
            .rename_axis("발화요인")
            .reset_index(name="발생건수")
        )

        fig1 = px.bar(
            cause_count,
            x="발화요인",
            y="발생건수",
            title="발화요인별 화재 발생 건수",
            color_discrete_sequence=[BLUE]
        )

        show_chart(fig1)

        # ----------------------------------
        # 건물구조별 발생 건수
        # ----------------------------------

        st.header("02. 건물구조별 화재 발생 현황")

        building_count = (
            filtered_df[BUILDING_COL]
            .value_counts()
            .head(10)
            .rename_axis("건물구조")
            .reset_index(name="발생건수")
        )

        fig2 = px.bar(
            building_count,
            x="건물구조",
            y="발생건수",
            title="건물구조별 화재 발생 건수 TOP 10",
            color_discrete_sequence=[BLUE]
        )

        show_chart(fig2)

        st.caption(
            "건물구조별 발생 건수는 시설 수를 "
            "보정한 화재 발생률이 아닙니다."
        )

        # ==================================
        # 03. 건물구조별 피해 심각도
        # ==================================

        st.header("03. 건물구조별 피해 심각도")

        st.caption(
            "현재 사이드바 필터가 적용된 데이터를 "
            "기준으로 피해 심각도를 분석합니다."
        )

        min_samples = st.slider(
            "최소 화재 발생 건수",
            min_value=1,
            max_value=500,
            value=100,
            step=1,
            key="building_min_samples"
        )

        building_df = filtered_df.copy()

        building_df["대형피해발생"] = (
                building_df[DAMAGE_COL] >= threshold_90
        )

        building_df["인명피해발생"] = (
                building_df["인명피해(명)"] > 0
        )

        building_risk = (
            building_df.groupby(BUILDING_COL)
            .agg(
                발생건수=("조사서번호", "size"),
                중앙값재산피해=(DAMAGE_COL, "median"),
                대형피해비율=("대형피해발생", "mean"),
                인명피해발생비율=("인명피해발생", "mean")
            )
            .reset_index()
        )

        building_risk["대형피해비율"] *= 100
        building_risk["인명피해발생비율"] *= 100

        building_risk = building_risk[
            building_risk["발생건수"] >= min_samples
            ].copy()

        if building_risk.empty:
            st.warning(
                "현재 조건에서 최소 표본 수를 "
                "충족하는 건물구조가 없습니다. "
                "슬라이더 값을 낮춰주세요."
            )

        else:
            metric_options = {
                "대형 피해 발생 비율 (%)": "대형피해비율",
                "인명피해 발생 비율 (%)": "인명피해발생비율",
                "재산피해 중앙값 (천 원)": "중앙값재산피해"
            }

            selected_metric = st.selectbox(
                "비교할 피해 지표",
                options=list(metric_options.keys()),
                key="building_risk_metric"
            )

            metric_col = metric_options[selected_metric]

            building_risk = building_risk.sort_values(
                metric_col,
                ascending=True
            )

            fig_risk = px.bar(
                building_risk,
                x=metric_col,
                y=BUILDING_COL,
                orientation="h",
                title=f"건물구조별 {selected_metric}",
                color_discrete_sequence=[BLUE],
                hover_data=["발생건수"]
            )

            show_chart(fig_risk)

            st.dataframe(
                building_risk.round(2),
                width="stretch"
            )

            st.caption(
                f"대형 피해 기준은 전체 데이터의 "
                f"90백분위({threshold_90:,.0f}천 원 이상)입니다. "
                "최소 표본 수는 탐색적 비교를 위한 설정입니다."
            )

        # ----------------------------------
        # 분석 데이터
        # ----------------------------------

        st.header("04. 분석 데이터 조회")

        st.dataframe(
            filtered_df,
            width="stretch"
        )

        st.download_button(
            "📥 필터링 데이터 CSV 다운로드",
            data=filtered_df.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="fire_insight_filtered.csv",
            mime="text/csv"
        )


# ==========================================
# TAB 2. 재산피해 분석
# ==========================================

with tab2:

    st.header("01. 발화요인별 재산피해 비교")

    if filtered_df.empty:
        st.info("표시할 재산피해 데이터가 없습니다.")

    else:
        damage_stats = (
            filtered_df.groupby("발화요인")[DAMAGE_COL]
            .agg(
                평균피해="mean",
                중앙값피해="median"
            )
            .reset_index()
        )

        damage_long = damage_stats.melt(
            id_vars="발화요인",
            value_vars=["평균피해", "중앙값피해"],
            var_name="통계",
            value_name="피해액(천원)"
        )

        fig3 = px.bar(
            damage_long,
            x="발화요인",
            y="피해액(천원)",
            color="통계",
            barmode="group",
            title="발화요인별 평균·중앙값 재산피해",
            color_discrete_map={
                "평균피해": BLUE,
                "중앙값피해": GRAY
            }
        )

        show_chart(fig3)

        st.info(
            "일부 대형 화재가 평균 피해액에 "
            "큰 영향을 주므로 중앙값도 함께 비교합니다."
        )

        st.header("02. 대형 피해 발생 비율")

        damage_df = filtered_df.copy()

        damage_df["대형피해"] = (
                damage_df[DAMAGE_COL] >= threshold_90
        )

        large_stats = (
            damage_df.groupby("발화요인")
            .agg(
                발생건수=("조사서번호", "size"),
                대형피해비율=("대형피해", "mean")
            )
            .reset_index()
        )

        large_stats["대형피해비율"] *= 100

        fig4 = px.bar(
            large_stats,
            x="발화요인",
            y="대형피해비율",
            hover_data=["발생건수"],
            title="발화요인별 대형 피해 발생 비율",
            color_discrete_sequence=[BLUE]
        )

        show_chart(fig4)

        st.caption(
            f"대형 피해 기준: 전체 데이터의 "
            f"90백분위({threshold_90:,.0f}천 원 이상). "
            "공식 대형 화재 분류 기준은 아닙니다."
        )

        st.header("03. 연소확대범위별 재산피해")

        spread_stats = (
            filtered_df.groupby("연소확대범위")[DAMAGE_COL]
            .median()
            .reset_index(name="중앙값피해")
        )

        fig5 = px.bar(
            spread_stats,
            x="연소확대범위",
            y="중앙값피해",
            title="연소확대범위별 재산피해 중앙값",
            color_discrete_sequence=[BLUE]
        )

        show_chart(fig5)

        st.caption(
            "연소확대범위와 피해 규모 사이의 "
            "관측된 연관성을 나타냅니다."
        )


# ==========================================
# TAB 3. 인명피해 분석
# ==========================================

with tab3:

    st.header("01. 발화요인별 인명피해 분석")

    if filtered_df.empty:
        st.info("표시할 인명피해 데이터가 없습니다.")

    else:
        casualty_df = filtered_df.copy()

        casualty_df["인명피해발생"] = (
                casualty_df["인명피해(명)"] > 0
        )

        casualty_stats = (
            casualty_df.groupby("발화요인")
            .agg(
                발생건수=("조사서번호", "size"),
                인명피해발생건수=("인명피해발생", "sum"),
                총사망자=("사망", "sum"),
                총부상자=("부상", "sum")
            )
            .reset_index()
        )

        casualty_stats["인명피해발생비율(%)"] = (
                casualty_stats["인명피해발생건수"]
                / casualty_stats["발생건수"] * 100
        ).round(2)

        casualty_plot = casualty_stats[
            casualty_stats["발생건수"] >= 100
            ].sort_values("인명피해발생비율(%)")

        if casualty_plot.empty:
            st.info(
                "현재 필터에서 100건 이상인 "
                "발화요인이 없습니다."
            )

        else:
            fig6 = px.bar(
                casualty_plot,
                x="인명피해발생비율(%)",
                y="발화요인",
                orientation="h",
                title="발화요인별 인명피해 발생 비율",
                text="인명피해발생비율(%)",
                color_discrete_sequence=[BLUE]
            )

            fig6.update_traces(
                texttemplate="%{text:.2f}%"
            )

            show_chart(fig6)

        st.dataframe(
            casualty_stats,
            width="stretch"
        )

        st.caption(
            "인명피해 발생 비율은 사망자 또는 부상자가 "
            "1명 이상 발생한 화재의 비율입니다."
        )


# ==========================================
# TAB 4. 통계 검정
# ==========================================

with tab4:

    st.header("통계적 검정 결과")

    if STAT_PATH.exists():
        stat_df = load_report(STAT_PATH)

        st.dataframe(
            stat_df,
            width="stretch"
        )

        st.info(
            "p값이 작더라도 효과크기가 작다면 "
            "실질적인 연관성이 크다고 단정할 수 없습니다."
        )


        st.caption(
            "전체 산업시설 화재 데이터를 기준으로 "
            "계산한 결과입니다. 사이드바 필터는 "
            "통계 검정 결과에 적용되지 않습니다."
        )

        st.warning(
            "발화층의 괄호 표기를 지하층으로 해석한 "
            "규칙은 공식 코드 정의 확인 전까지 "
            "잠정적인 분류입니다."
        )

    else:
        st.error(
            "통계 검정 결과 CSV 파일이 없습니다. "
            "07_statistical_analysis.py를 실행해 주세요."
        )


# ==========================================
# TAB 5. 데이터 품질 및 전처리
# ==========================================

with tab5:

    st.header("01. 데이터 품질 및 결측치 분석")

    if QUALITY_PATH.exists():

        quality_df = load_report(QUALITY_PATH)

        st.dataframe(
            quality_df,
            width="stretch"
        )

        quality_plot = quality_df.sort_values(
            "결측률(%)",
            ascending=False
        )

        fig7 = px.bar(
            quality_plot,
            x="결측률(%)",
            y="컬럼명",
            orientation="h",
            title="컬럼별 결측률 (%)",
            color_discrete_sequence=[BLUE]
        )

        fig7.update_layout(
            yaxis={
                "categoryorder": "total ascending"
            }
        )

        show_chart(fig7)

        st.caption(
            "결측률은 산업시설 화재 데이터의 "
            "전처리 이전 품질 진단 결과입니다. "
            "결측값을 무조건 삭제하거나 "
            "0으로 대체하지 않았습니다."
        )

    else:
        st.error(
            "데이터 품질 보고서가 없습니다. "
            "02_preprocessing.py를 실행해 주세요."
        )

    # ======================================
    # 02. 데이터 전처리 현황
    # ======================================

    st.header("02. 데이터 전처리 현황")

    pre1, pre2, pre3 = st.columns(3)

    pre1.metric(
        "산업시설 추출 데이터",
        "24,678건"
    )

    pre2.metric(
        "전처리 후 데이터",
        f"{len(df):,}건"
    )

    pre3.metric(
        "전처리 후 컬럼",
        f"{len(df.columns)}개"
    )

    st.markdown("""
**주요 전처리 작업**

- 산업시설 화재 데이터 추출
- 조사서번호 중복 검사
- 재산피해 숫자형 검증 및 로그 변환
- 건물층수 결측 여부 표시
- 발화층 표기 정규화 및 범주화
- 건물구조 표준화 (`기타/미분류`)
- 소방설비 관련 결측 여부 컬럼 생성
- Parquet 형식으로 분석용 데이터 저장
""")

    st.warning(
        "발화층의 괄호 표기를 지하층으로 분류한 "
        "규칙은 공식 코드 정의 확인 전까지 "
        "잠정적인 해석입니다."
    )

    # ======================================
    # 03. 건물구조 표준화 검증
    # ======================================

    st.header("03. 건물구조 표준화 결과")

    original_count = df["건물구조조"].nunique()
    standardized_count = df[BUILDING_COL].nunique()

    col_a, col_b = st.columns(2)

    col_a.metric(
        "표준화 전 범주 수",
        f"{original_count}개"
    )

    col_b.metric(
        "표준화 후 범주 수",
        f"{standardized_count}개"
    )

    st.caption(
        "원본 건물구조 데이터는 보존하고 "
        "분석용 표준화 컬럼을 추가했습니다."
    )

    building_mapping = (
        df.groupby(["건물구조조", BUILDING_COL])
        .size()
        .reset_index(name="화재건수")
        .rename(columns={
            "건물구조조": "원본 건물구조",
            BUILDING_COL: "표준화 건물구조"
        })
        .sort_values(
            "화재건수",
            ascending=False
        )
    )

    st.dataframe(
        building_mapping,
        width="stretch"
    )

    st.caption(
        "기타 계열은 분석 편의를 위해 "
        "'기타/미분류'로 통합했습니다. "
        "공식 건축구조 분류의 동등성을 "
        "의미하지는 않습니다."
    )


# ==========================================
# 하단 정보
# ==========================================

st.divider()

st.caption(
    "Fire Insight 2.0 | "
    "소방청 공공데이터 기반 산업시설 화재 분석 프로젝트"
)
