from pathlib import Path
import pandas as pd

# =====================================
# 1. 파일 경로 설정
# =====================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data" / "raw" / "fire_20260831.csv"
OUTPUT_PATH = BASE_DIR / "data" / "interim" / "fire_industrial.csv"

# =====================================
# 2. 원본 데이터 불러오기
# =====================================

df = pd.read_csv(
    RAW_PATH,
    encoding="cp949",
    low_memory=False
)

print("========== 데이터 기본 정보 ==========")
print(f"전체 데이터: {len(df):,}건")
print(f"전체 컬럼: {len(df.columns)}개")

# =====================================
# 3. 컬럼 정보 확인
# =====================================

print("\n========== 컬럼 목록 ==========")

for index, column in enumerate(df.columns, start=1):
    print(f"{index}. {column}")

print("\n========== 자료형 ==========")
print(df.dtypes)

# =====================================
# 4. 결측치 분석
# =====================================

missing = pd.DataFrame({
    "결측 건수": df.isna().sum(),
    "결측률(%)": (df.isna().mean() * 100).round(2)
})

print("\n========== 결측치 분석 ==========")
print(missing.to_string())

# =====================================
# 5. 장소별 화재 건수
# =====================================

print("\n========== 장소별 화재 건수 ==========")
print(df["장소"].value_counts(dropna=False))

# =====================================
# 6. 산업시설 화재 추출
# =====================================

industrial_df = df[df["장소"] == "산업시설"].copy()

print("\n========== 산업시설 화재 분석 ==========")
print(f"산업시설 화재: {len(industrial_df):,}건")
print(f"전체 대비 비율: {len(industrial_df) / len(df) * 100:.2f}%")

# =====================================
# 7. 중복 데이터 검사
# =====================================

duplicate_count = industrial_df["조사서번호"].duplicated().sum()

print("\n========== 중복 검사 ==========")
print(f"조사서번호 중복: {duplicate_count:,}건")

# =====================================
# 8. 산업시설 데이터 저장
# =====================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

industrial_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== 저장 완료 ==========")
print(f"저장 위치: {OUTPUT_PATH}")