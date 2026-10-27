
from pathlib import Path

import pandas as pd

# =====================================
# 1. 경로 설정
# =====================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
        BASE_DIR / "data/processed/fire_industrial_clean.parquet"
)

OUTPUT_PATH = (
        BASE_DIR / "reports/building_risk_analysis.csv"
)

# =====================================
# 2. 데이터 불러오기
# =====================================

df = pd.read_parquet(INPUT_PATH)

BUILDING_COL = "건물구조_표준화"
DAMAGE_COL = "재산피해(천원)"

df[DAMAGE_COL] = pd.to_numeric(
    df[DAMAGE_COL],
    errors="coerce"
)

# =====================================
# 3. 분석용 변수 생성
# =====================================

# 전체 산업시설 기준 90백분위
threshold_90 = df[DAMAGE_COL].quantile(0.90)

df["대형피해발생"] = (
        df[DAMAGE_COL] >= threshold_90
)

df["인명피해발생"] = (
        df["인명피해(명)"] > 0
)

# =====================================
# 4. 건물구조별 피해 통계
# =====================================

building_stats = (
    df.groupby(BUILDING_COL)
    .agg(
        발생건수=("조사서번호", "size"),
        평균재산피해=(DAMAGE_COL, "mean"),
        중앙값재산피해=(DAMAGE_COL, "median"),
        대형피해발생건수=("대형피해발생", "sum"),
        인명피해발생건수=("인명피해발생", "sum")
    )
)

building_stats["대형피해비율(%)"] = (
        building_stats["대형피해발생건수"]
        / building_stats["발생건수"] * 100
)

building_stats["인명피해발생비율(%)"] = (
        building_stats["인명피해발생건수"]
        / building_stats["발생건수"] * 100
)

building_stats = building_stats.round(2)

# =====================================
# 5. 표본 수 기준 적용
# =====================================

# 분석 대상은 100건 이상인 건물구조
reliable_stats = building_stats[
    building_stats["발생건수"] >= 100
    ].copy()

reliable_stats = reliable_stats.sort_values(
    "대형피해비율(%)",
    ascending=False
)

# =====================================
# 6. 결과 출력
# =====================================

print("========== 건물구조별 피해 분석 ==========")
print(f"전체 화재: {len(df):,}건")
print(f"전체 건물구조: {len(building_stats)}개")
print(f"100건 이상 건물구조: {len(reliable_stats)}개")

print("\n========== 주요 건물구조별 피해 ==========")
print(reliable_stats.to_string())

# =====================================
# 7. 결과 저장
# =====================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

building_stats.to_csv(
    OUTPUT_PATH,
    encoding="utf-8-sig"
)

print("\n분석 결과 저장 완료")
print(OUTPUT_PATH)
