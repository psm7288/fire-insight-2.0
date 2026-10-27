
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 경로 설정 및 데이터 불러오기
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data/processed/fire_industrial_clean.parquet"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT_PATH)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 2. 인명피해 데이터 검증
# ==========================================

for col in ["인명피해(명)", "사망", "부상", "재산피해(천원)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 사망 + 부상 = 인명피해 검증
valid = df[["인명피해(명)", "사망", "부상"]].notna().all(axis=1)

mismatch = (
        df.loc[valid, "사망"] + df.loc[valid, "부상"]
        != df.loc[valid, "인명피해(명)"]
).sum()

print("========== 인명피해 데이터 검증 ==========")
print(f"전체 화재: {len(df):,}건")
print(f"인명피해 합계 불일치: {mismatch:,}건")

# 인명피해가 발생한 화재인지 구분
df["인명피해발생"] = df["인명피해(명)"] > 0
df["사망발생"] = df["사망"] > 0
df["부상발생"] = df["부상"] > 0

print("\n========== 인명피해 현황 ==========")
print(f"인명피해 발생 화재: {df['인명피해발생'].sum():,}건")
print(f"인명피해 발생 비율: {df['인명피해발생'].mean()*100:.2f}%")
print(f"총 사망자: {df['사망'].sum():,.0f}명")
print(f"총 부상자: {df['부상'].sum():,.0f}명")

# ==========================================
# 3. 발화요인별 인명피해 분석
# ==========================================

cause_stats = (
    df.groupby("발화요인")
    .agg(
        발생건수=("조사서번호", "size"),
        인명피해발생건수=("인명피해발생", "sum"),
        사망발생건수=("사망발생", "sum"),
        총사망자=("사망", "sum"),
        총부상자=("부상", "sum")
    )
)

cause_stats["인명피해발생비율(%)"] = (
        cause_stats["인명피해발생건수"]
        / cause_stats["발생건수"] * 100
).round(2)

cause_stats = cause_stats.sort_values(
    "발생건수", ascending=False
)

print("\n========== 발화요인별 인명피해 ==========")
print(cause_stats.to_string())

cause_stats.to_csv(
    REPORT_DIR / "casualty_by_cause.csv",
    encoding="utf-8-sig"
)

# ==========================================
# 4. 건물구조별 피해 분석
# ==========================================

building_stats = (
    df.groupby("건물구조조")
    .agg(
        발생건수=("조사서번호", "size"),
        인명피해발생건수=("인명피해발생", "sum"),
        평균재산피해=("재산피해(천원)", "mean"),
        중앙값재산피해=("재산피해(천원)", "median")
    )
)

building_stats["인명피해발생비율(%)"] = (
        building_stats["인명피해발생건수"]
        / building_stats["발생건수"] * 100
).round(2)

building_stats = building_stats.sort_values(
    "발생건수", ascending=False
).round(2)

print("\n========== 건물구조별 피해 ==========")
print(building_stats.head(15).to_string())

building_stats.to_csv(
    REPORT_DIR / "damage_by_building.csv",
    encoding="utf-8-sig"
)

# ==========================================
# 5. 발화층별 피해 분석
# ==========================================

floor_stats = (
    df.groupby("발화층_구분")
    .agg(
        발생건수=("조사서번호", "size"),
        인명피해발생건수=("인명피해발생", "sum"),
        평균재산피해=("재산피해(천원)", "mean"),
        중앙값재산피해=("재산피해(천원)", "median")
    )
)

floor_stats["인명피해발생비율(%)"] = (
        floor_stats["인명피해발생건수"]
        / floor_stats["발생건수"] * 100
).round(2)

print("\n========== 발화층별 피해 ==========")
print(floor_stats.round(2).to_string())

floor_stats.to_csv(
    REPORT_DIR / "damage_by_floor.csv",
    encoding="utf-8-sig"
)

# ==========================================
# 6. 연소확대범위별 피해 분석
# ==========================================

spread_stats = (
    df.groupby("연소확대범위")
    .agg(
        발생건수=("조사서번호", "size"),
        인명피해발생건수=("인명피해발생", "sum"),
        평균재산피해=("재산피해(천원)", "mean"),
        중앙값재산피해=("재산피해(천원)", "median")
    )
)

spread_stats["인명피해발생비율(%)"] = (
        spread_stats["인명피해발생건수"]
        / spread_stats["발생건수"] * 100
).round(2)

spread_stats = spread_stats.sort_values(
    "발생건수", ascending=False
).round(2)

print("\n========== 연소확대범위별 피해 ==========")
print(spread_stats.to_string())

spread_stats.to_csv(
    REPORT_DIR / "damage_by_spread.csv",
    encoding="utf-8-sig"
)

# ==========================================
# 7. 그래프 생성
# ==========================================

# 그래프 1: 발화요인별 인명피해 발생 비율
# 표본 수 100건 이상인 요인만 비교
plot_cause = cause_stats[
    cause_stats["발생건수"] >= 100
    ].sort_values("인명피해발생비율(%)")

plt.figure(figsize=(11, 6))
plt.barh(
    plot_cause.index,
    plot_cause["인명피해발생비율(%)"]
)
plt.title("발화요인별 인명피해 발생 비율 (100건 이상)")
plt.xlabel("인명피해 발생 화재 비율 (%)")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "casualty_rate_by_cause.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# 그래프 2: 주요 건물구조별 화재 발생 건수
plot_building = building_stats.head(10).sort_values("발생건수")

plt.figure(figsize=(11, 6))
plt.barh(
    plot_building.index,
    plot_building["발생건수"]
)
plt.title("건물구조별 산업시설 화재 발생 건수")
plt.xlabel("화재 발생 건수")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "fire_count_by_building.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# 그래프 3: 연소확대범위별 중앙값 재산피해
plot_spread = spread_stats.sort_values("중앙값재산피해")

plt.figure(figsize=(11, 6))
plt.barh(
    plot_spread.index,
    plot_spread["중앙값재산피해"]
)
plt.title("연소확대범위별 재산피해 중앙값")
plt.xlabel("중앙값 재산피해 (천 원)")
plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "damage_by_spread.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("\n========== 6단계 분석 완료 ==========")
print("CSV 보고서 4개 생성")
print("PNG 그래프 3개 생성")
