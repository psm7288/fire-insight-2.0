
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =====================================
# 1. 페이지 설정
# =====================================

st.set_page_config(
    page_title="Fire Insight 2.0",
    page_icon="🔥",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"

BLUE = "#2563EB"
GRAY = "#94A3B8"


# =====================================
# 2. 데이터 로딩
# =====================================

@st.cache_data
def load_report(filename):
    path = REPORT_DIR / filename

    if not path.exists():
        return None

    return pd.read_csv(
        path,
        encoding="utf-8-sig"
    )


def show_chart(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(size=13),
        margin=dict(l=20, r=20, t=60, b=30)
    )
    st.plotly_chart(fig, width="stretch")


def format_money(value):
    if pd.isna(value):
        return "자료 없음"

    won = int(round(value * 1000))

    if abs(won) >= 100_000_000:
        return f"{won / 100_000_000:,.2f}억 원"

    if abs(won) >= 10_000:
        return f"{won / 10_000:,.0f}만 원"

    return f"{won:,}원"


cause_df = load_report("fire_cause_count.csv")
damage_df = load_report("fire_damage_advanced.csv")
casualty_df = load_report("casualty_by_cause.csv")
building_df = load_report("building_risk_analysis.csv")
quality_df = load_report("data_quality.csv")
summary_df = load_report("preprocessing_summary.csv")
validation_df = load_report("preprocessing_validation.csv")
stat_df = load_report("statistical_test_results.csv")


# =====================================
# 3. 메인 화면
# =====================================

st.title("🔥 Fire Insight 2.0")
st.subheader("산업시설 화재 발생 특성 및 피해 심각도 분석")

st.caption(
    "소방청 공공데이터 기반 산업시설 화재 분석 | "
    "집계 데이터 공개용 대시보드"
)

if cause_df is None or damage_df is None:
    st.error("필수 분석 보고서가 없습니다.")
    st.stop()

total_count = int(cause_df["발생건수"].sum())

# 상위 1% 피해 점유율은 기존 전체 분석 결과
# 현재 공개용 보고서에는 별도 저장되지 않았으므로
# 재계산하지 않고 표시 여부를 구분한다.

k1, k2, k3 = st.columns(3)

k1.metric("산업시설 화재", f"{total_count:,}건")
k2.metric("발화요인 종류", f"{len(cause_df):,}개")
k3.metric(
    "건물구조 분류",
    f"{len(building_df):,}개" if building_df is not None else "자료 없음"
)

st.divider()

tabs = st.tabs([
    "📊 종합 현황",
    "💰 재산피해",
    "🚑 인명피해",
    "🏭 건물구조",
    "🗂️ 데이터 품질",
    "📈 통계 검정"
])

# =====================================
# TAB 1. 종합 현황
# =====================================

with tabs[0]:
    st.header("발화요인별 화재 발생 현황")

    fig = px.bar(
        cause_df,
        x="발화요인",
        y="발생건수",
        title="발화요인별 화재 발생 건수",
        color_discrete_sequence=[BLUE]
    )
    show_chart(fig)

    st.dataframe(cause_df, width="stretch")

    st.info(
        "전기적 요인과 부주의가 주요 발화요인으로 "
        "나타났습니다. 발생 건수는 산업시설별 "
        "노출량을 보정한 화재 위험률이 아닙니다."
    )


# =====================================
# TAB 2. 재산피해
# =====================================

with tabs[1]:
    st.header("발화요인별 재산피해 심각도")

    if "발화요인" not in damage_df.columns:
        damage_df = damage_df.rename(
            columns={damage_df.columns[0]: "발화요인"}
        )

    st.dataframe(damage_df, width="stretch")

    metric_options = {
        "평균 재산피해": "평균피해",
        "중앙값 재산피해": "중앙값피해",
        "90백분위 재산피해": "피해90백분위",
        "대형 피해 비율": "대형피해비율(%)"
    }

    metric_label = st.selectbox(
        "비교 지표",
        list(metric_options.keys())
    )

    metric_col = metric_options[metric_label]

    plot_df = damage_df.sort_values(
        metric_col,
        ascending=False
    )

    fig = px.bar(
        plot_df,
        x="발화요인",
        y=metric_col,
        title=f"발화요인별 {metric_label}",
        color_discrete_sequence=[BLUE]
    )
    show_chart(fig)

    st.caption(
        "대형 피해는 전체 산업시설 화재의 "
        "재산피해 90백분위 이상으로 정의했습니다. "
        "이는 분석용 상대적 기준입니다."
    )


# =====================================
# TAB 3. 인명피해
# =====================================

with tabs[2]:
    st.header("발화요인별 인명피해")

    if casualty_df is not None:

        if "발화요인" not in casualty_df.columns:
            casualty_df = casualty_df.rename(
                columns={casualty_df.columns[0]: "발화요인"}
            )

        plot_df = casualty_df[
            casualty_df["발생건수"] >= 100
            ].sort_values("인명피해발생비율(%)")

        fig = px.bar(
            plot_df,
            x="인명피해발생비율(%)",
            y="발화요인",
            orientation="h",
            title="발화요인별 인명피해 발생 비율",
            color_discrete_sequence=[BLUE]
        )
        show_chart(fig)

        st.dataframe(casualty_df, width="stretch")

    else:
        st.warning("인명피해 보고서가 없습니다.")


# =====================================
# TAB 4. 건물구조
# =====================================

with tabs[3]:
    st.header("건물구조별 피해 심각도")

    if building_df is not None:

        if "건물구조" not in building_df.columns:
            building_df = building_df.rename(
                columns={building_df.columns[0]: "건물구조"}
            )

        building_plot = building_df[
            building_df["발생건수"] >= 100
            ].copy()

        metric_options = {
            "대형 피해 발생 비율": "대형피해비율(%)",
            "인명피해 발생 비율": "인명피해발생비율(%)",
            "재산피해 중앙값": "중앙값재산피해"
        }

        selected_metric = st.selectbox(
            "건물구조 비교 지표",
            list(metric_options.keys())
        )

        metric_col = metric_options[selected_metric]

        building_plot = building_plot.sort_values(
            metric_col,
            ascending=True
        )

        fig = px.bar(
            building_plot,
            x=metric_col,
            y="건물구조",
            orientation="h",
            title=f"건물구조별 {selected_metric}",
            color_discrete_sequence=[BLUE]
        )
        show_chart(fig)

        st.dataframe(building_df, width="stretch")

        st.caption(
            "발생 건수 100건 이상인 건물구조를 "
            "그래프에 표시합니다. 시설 수나 규모를 "
            "보정한 위험률은 아닙니다."
        )

    else:
        st.warning("건물구조 분석 보고서가 없습니다.")


# =====================================
# TAB 5. 데이터 품질
# =====================================

with tabs[4]:
    st.header("데이터 품질 및 전처리")

    if quality_df is not None:
        st.dataframe(quality_df, width="stretch")

        fig = px.bar(
            quality_df.sort_values("결측률(%)"),
            x="결측률(%)",
            y="컬럼명",
            orientation="h",
            title="컬럼별 결측률",
            color_discrete_sequence=[BLUE]
        )
        show_chart(fig)

    if summary_df is not None:
        st.subheader("전처리 전후 비교")
        st.dataframe(summary_df, width="stretch")

    if validation_df is not None:
        st.subheader("전처리 검증 결과")
        st.dataframe(validation_df, width="stretch")

    st.caption(
        "원본 19개 컬럼을 전처리하여 "
        "27개 분석용 컬럼을 구성했습니다. "
        "데이터 24,678건은 모두 보존했습니다."
    )


# =====================================
# TAB 6. 통계 검정
# =====================================

with tabs[5]:
    st.header("통계적 검정 결과")

    if stat_df is not None:
        st.dataframe(stat_df, width="stretch")

        st.info(
            "p값과 효과크기를 함께 고려해야 하며, "
            "통계적 연관성을 인과관계로 해석하지 않습니다."
        )
    else:
        st.warning("통계 검정 보고서가 없습니다.")


# =====================================
# 하단 정보
# =====================================

st.divider()

st.caption(
    "Fire Insight 2.0 | "
    "소방청 공공데이터 기반 산업시설 화재 분석"
)
