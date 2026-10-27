from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. 경로 설정
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
        BASE_DIR / "data/processed/fire_industrial_clean.parquet"
)

FIGURE_DIR = BASE_DIR / "reports/figures"
REPORT_DIR = BASE_DIR / "reports"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# 2. 전처리 데이터 불러오기
df = pd.read_parquet(INPUT_PATH)

print("========== 분석 데이터 ==========")
print(f"전체 산업시설 화재: {len(df):,}건")

# 3. 발화요인별 화재 발생 건수
cause_count = (
    df["발화요인"]
    .value_counts()
    .rename_axis("발화요인")
    .reset_index(name="발생건수")
)

cause_count["발생비율(%)"] = (
        cause_count["발생건수"] / len(df) * 100
).round(2)

print("\n========== 발화요인별 발생 현황 ==========")
print(cause_count.to_string(index=False))

# 4. 발화요인별 재산피해 통계
damage_stats = (
    df.groupby("발화요인")["재산피해(천원)"]
    .agg(
        발생건수="count",
        평균피해="mean",
        중앙값피해="median",
        최대피해="max"
    )
    .round(2)
    .sort_values("발생건수", ascending=False)
)

print("\n========== 발화요인별 재산피해 ==========")
print(damage_stats.to_string())

# 5. 통계 결과 CSV 저장
cause_count.to_csv(
    REPORT_DIR / "fire_cause_count.csv",
    index=False,
    encoding="utf-8-sig"
)

damage_stats.to_csv(
    REPORT_DIR / "fire_damage_stats.csv",
    encoding="utf-8-sig"
)

# 6. 한글 폰트 설정 (macOS)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 7. 발화요인별 발생 건수 그래프
top_causes = cause_count.head(10).sort_values("발생건수")

plt.figure(figsize=(11, 6))

plt.barh(
    top_causes["발화요인"],
    top_causes["발생건수"]
)

plt.title("산업시설 화재 발화요인별 발생 건수")
plt.xlabel("화재 발생 건수")
plt.ylabel("발화요인")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "fire_cause_count.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\n========== 분석 완료 ==========")
print("발화요인별 통계 및 그래프 저장 완료")
