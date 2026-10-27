from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================
# 1. 경로 설정 및 데이터 불러오기
# =====================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data/processed/fire_industrial_clean.parquet"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT_PATH)

# 재산피해 단위: 천 원
damage_col = "재산피해(천원)"
df[damage_col] = pd.to_numeric(df[damage_col], errors="coerce")

valid_df = df[df[damage_col].notna() & (df[damage_col] >= 0)].copy()

print("========== 분석 데이터 ==========")
print(f"전체 데이터: {len(df):,}건")
print(f"유효 피해 데이터: {len(valid_df):,}건")

# =====================================
# 2. 재산피해 기초 통계
# =====================================

damage = valid_df[damage_col]

print("\n========== 재산피해 통계 ==========")
print(damage.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

print(f"평균 피해: {damage.mean():,.2f}천 원")
print(f"중앙값 피해: {damage.median():,.2f}천 원")

# =====================================
# 3. 대형 피해 기준 설정
# =====================================

# 전체 산업시설 화재의 90백분위를 분석 기준으로 사용
threshold_90 = damage.quantile(0.90)
threshold_99 = damage.quantile(0.99)

valid_df["대형피해여부"] = valid_df[damage_col] >= threshold_90
valid_df["상위1퍼센트여부"] = valid_df[damage_col] >= threshold_99

print("\n========== 피해 기준 ==========")
print(f"대형 피해 기준(90백분위): {threshold_90:,.2f}천 원")
print(f"상위 1% 기준(99백분위): {threshold_99:,.2f}천 원")

# =====================================
# 4. 발화요인별 피해 통계
# =====================================

cause_stats = (
    valid_df.groupby("발화요인")[damage_col]
    .agg(
        발생건수="count",
        평균피해="mean",
        중앙값피해="median",
        최대피해="max",
        피해90백분위=lambda x: x.quantile(0.90)
    )
)

large_damage = (
        valid_df.groupby("발화요인")["대형피해여부"]
        .mean() * 100
)

cause_stats["대형피해비율(%)"] = large_damage.round(2)

cause_stats = cause_stats.sort_values(
    "발생건수", ascending=False
).round(2)

print("\n========== 발화요인별 피해 심각도 ==========")
print(cause_stats.to_string())

cause_stats.to_csv(
    REPORT_DIR / "fire_damage_advanced.csv",
    encoding="utf-8-sig"
)

# =====================================
# 5. 상위 1% 재산피해 집중도
# =====================================

top_1_damage = valid_df.loc[
    valid_df["상위1퍼센트여부"], damage_col
].sum()

total_damage = damage.sum()

top_1_share = (
    top_1_damage / total_damage * 100
    if total_damage > 0 else np.nan
)

print("\n========== 상위 1% 피해 집중도 ==========")
print(f"전체 재산피해: {total_damage:,.0f}천 원")
print(f"상위 1% 사건 피해 합계: {top_1_damage:,.0f}천 원")
print(f"상위 1% 피해 비중: {top_1_share:.2f}%")

# =====================================
# 6. 그래프 설정
# =====================================

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 그래프 1: 재산피해 로그 분포
plt.figure(figsize=(10, 6))

plt.hist(
    np.log1p(damage),
    bins=50,
    edgecolor="black"
)

plt.title("산업시설 화재 재산피해 분포")
plt.xlabel("log(1 + 재산피해[천 원])")
plt.ylabel("화재 건수")

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "damage_distribution.png",
    dpi=200
)
plt.close()

# 그래프 2: 발화요인별 중앙값 피해
top_causes = cause_stats.head(8).sort_values("중앙값피해")

plt.figure(figsize=(11, 6))

plt.barh(
    top_causes.index,
    top_causes["중앙값피해"]
)

plt.title("발화요인별 재산피해 중앙값")
plt.xlabel("중앙값 피해액(천 원)")

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "damage_median_by_cause.png",
    dpi=200
)
plt.close()

# 그래프 3: 발화요인별 대형 피해 비율
top_causes = cause_stats.head(8).sort_values("대형피해비율(%)")

plt.figure(figsize=(11, 6))

plt.barh(
    top_causes.index,
    top_causes["대형피해비율(%)"]
)

plt.title("발화요인별 대형 피해 발생 비율")
plt.xlabel("대형 피해 발생 비율(%)")

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "large_damage_rate.png",
    dpi=200
)
plt.close()

print("\n========== 분석 완료 ==========")
print("재산피해 심화 분석 완료")
print("통계 저장: reports/fire_damage_advanced.csv")
print("그래프 3개 저장 완료")