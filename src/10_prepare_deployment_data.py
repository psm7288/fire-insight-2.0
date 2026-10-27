from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
        BASE_DIR / "data/processed/fire_industrial_clean.parquet"
)

OUTPUT_PATH = (
        BASE_DIR / "data/deploy/fire_dashboard.parquet"
)

df = pd.read_parquet(INPUT_PATH)

# 대시보드에 필요한 컬럼만 선택
public_columns = [
    "발화요인",
    "건물구조_표준화",
    "재산피해(천원)",
    "인명피해(명)",
    "사망",
    "부상",
    "연소확대범위"
]

deploy_df = df[public_columns].copy()

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

deploy_df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("========== 배포용 데이터 ==========")
print(f"데이터 건수: {len(deploy_df):,}")
print(f"컬럼 수: {len(deploy_df.columns)}")
print(f"저장 위치: {OUTPUT_PATH}")