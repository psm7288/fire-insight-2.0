from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data/interim/fire_industrial.csv"
OUTPUT_PATH = BASE_DIR / "data/processed/fire_industrial_clean.parquet"

df = pd.read_csv(
    INPUT_PATH,
    encoding="utf-8-sig",
    dtype={"조사서번호": "string"},
    low_memory=False
)

original_count = len(df)

# 1. 식별자 정리
df["조사서번호"] = df["조사서번호"].str.strip()

# 2. 재산피해 숫자형 변환
df["재산피해(천원)"] = pd.to_numeric(
    df["재산피해(천원)"],
    errors="coerce"
)

# 3. 재산피해 로그 변환
df["재산피해_log"] = np.log1p(
    df["재산피해(천원)"]
)

# 4. 건물층수 숫자형 변환
for col in ["건물층수(지상)", "건물층수(지하)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col + "_결측"] = df[col].isna()

# 5. 발화층 데이터 전처리

import re

def classify_floor(value):
    value = str(value).strip()

    # 결측값
    if value in ["", "nan", "None"]:
        return "미상"

    # 괄호로 표시된 층수: 지하층 추정
    if re.fullmatch(r"\(\d+\)", value):
        return "지하"

    # 숫자로 표시된 층수: 지상층
    if re.fullmatch(r"\d+", value):
        return "지상"

    return "미분류"


df["발화층_구분"] = df["발화층"].apply(classify_floor)

# 6. 설비 및 초기소화활동 결측 표시
for col in ["자동소화설비", "자동화재탐지설비", "초기소화활동"]:
    df[col + "_미기록"] = df[col].isna()

# 6-1. 건물구조 표준화

building_map = {
    "기타": "기타/미분류",
    "기타 조": "기타/미분류",
    "기타(건물구조조코드)": "기타/미분류",
    "블럭조": "블록조"
}

df["건물구조_표준화"] = (
    df["건물구조조"]
    .astype("string")
    .str.strip()
    .replace(building_map)
)

print("\n========== 건물구조 표준화 ==========")
print("원본 범주 수:", df["건물구조조"].nunique())
print("표준화 범주 수:", df["건물구조_표준화"].nunique())

print("\n표준화 후 건물구조 분포:")
print(df["건물구조_표준화"].value_counts())

# 7. 저장
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("========== 전처리 결과 ==========")
print(f"전처리 전: {original_count:,}건")
print(f"전처리 후: {len(df):,}건")
print(f"컬럼 수: {len(df.columns)}개")

print("\n발화층 구분:")
print(df["발화층_구분"].value_counts(dropna=False))

print("\n저장 완료:", OUTPUT_PATH)