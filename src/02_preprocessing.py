from pathlib import Path
import pandas as pd

# 1. 경로 설정
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data/interim/fire_industrial.csv"
REPORT_PATH = BASE_DIR / "reports/data_quality.csv"

# 2. 산업시설 데이터 불러오기
df = pd.read_csv(
    INPUT_PATH,
    encoding="utf-8-sig",
    low_memory=False
)

print("========== 산업시설 데이터 ==========")
print(f"전체 데이터: {len(df):,}건")
print(f"전체 컬럼: {len(df.columns)}개")

# 3. 결측치 및 자료형 분석
quality = pd.DataFrame({
    "자료형": df.dtypes.astype(str),
    "결측 건수": df.isna().sum(),
    "결측률(%)": (df.isna().mean() * 100).round(2),
    "고유값 수": df.nunique(dropna=True)
})

quality.index.name = "컬럼명"

print("\n========== 데이터 품질 분석 ==========")
print(quality.to_string())

# 4. 수치형 데이터 통계
print("\n========== 수치형 데이터 통계 ==========")
print(df.describe().to_string())

# 5. 범주형 데이터 분포
for col in ["발화요인", "화재유형", "건물구조조"]:
    if col in df.columns:
        print(f"\n========== {col} ==========")
        print(df[col].value_counts(dropna=False).head(20))

# 6. 재산피해 이상치 후보 확인
damage = pd.to_numeric(
    df["재산피해(천원)"],
    errors="coerce"
)

print("\n========== 재산피해 분석 ==========")
print(damage.describe(
    percentiles=[0.5, 0.9, 0.95, 0.99]
))

print("음수 피해액:", (damage < 0).sum())
print("숫자 변환 실패:", (
        damage.isna() & df["재산피해(천원)"].notna()
).sum())

# 7. 품질 보고서 저장
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

quality.to_csv(
    REPORT_PATH,
    encoding="utf-8-sig"
)

print("\n데이터 품질 보고서 저장 완료")
print(REPORT_PATH)