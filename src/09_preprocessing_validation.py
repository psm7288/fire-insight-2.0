
from pathlib import Path
import pandas as pd

# =====================================
# 1. 경로 설정
# =====================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data/interim/fire_industrial.csv"
CLEAN_PATH = BASE_DIR / "data/processed/fire_industrial_clean.parquet"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# 2. 데이터 불러오기
# =====================================

raw = pd.read_csv(
    RAW_PATH,
    encoding="utf-8-sig",
    dtype={"조사서번호": "string"},
    low_memory=False
)

clean = pd.read_parquet(CLEAN_PATH)

raw["조사서번호"] = raw["조사서번호"].str.strip()
clean["조사서번호"] = clean["조사서번호"].astype("string").str.strip()

# =====================================
# 3. 기본 검증
# =====================================

raw_count = len(raw)
clean_count = len(clean)

retention_rate = (
    clean_count / raw_count * 100
    if raw_count else 0
)

raw_ids = set(raw["조사서번호"].dropna())
clean_ids = set(clean["조사서번호"].dropna())

checks = {
    "행 수 일치": raw_count == clean_count,
    "조사서번호 집합 일치": raw_ids == clean_ids,
    "원본 조사서번호 중복 없음": not raw["조사서번호"].duplicated().any(),
    "전처리 조사서번호 중복 없음": not clean["조사서번호"].duplicated().any(),
    "건물구조 표준화 컬럼 존재": "건물구조_표준화" in clean.columns,
    "재산피해 로그 컬럼 존재": "재산피해_log" in clean.columns
}

# =====================================
# 4. 건물구조 표준화 검증
# =====================================

building_mapping = {
    "기타": "기타/미분류",
    "기타 조": "기타/미분류",
    "기타(건물구조조코드)": "기타/미분류",
    "블럭조": "블록조"
}

expected_building = (
    clean["건물구조조"]
    .astype("string")
    .str.strip()
    .replace(building_mapping)
)

checks["건물구조 표준화 일치"] = (
    expected_building.equals(
        clean["건물구조_표준화"].astype("string")
    )
)

# =====================================
# 5. 전처리 결과 요약
# =====================================

summary = pd.DataFrame([
    {
        "지표": "데이터 건수",
        "전처리 전": raw_count,
        "전처리 후": clean_count
    },
    {
        "지표": "컬럼 수",
        "전처리 전": len(raw.columns),
        "전처리 후": len(clean.columns)
    },
    {
        "지표": "건물구조 범주 수",
        "전처리 전": raw["건물구조조"].nunique(),
        "전처리 후": clean["건물구조_표준화"].nunique()
    }
])

summary.to_csv(
    REPORT_DIR / "preprocessing_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================
# 6. 검증 결과 저장
# =====================================

validation = pd.DataFrame([
    {"검증 항목": name, "통과 여부": bool(result)}
    for name, result in checks.items()
])

validation.to_csv(
    REPORT_DIR / "preprocessing_validation.csv",
    index=False,
    encoding="utf-8-sig"
)

print("========== 전처리 검증 ==========")
print(f"전처리 전: {raw_count:,}건")
print(f"전처리 후: {clean_count:,}건")
print(f"데이터 보존율: {retention_rate:.2f}%")

print("\n========== 검증 항목 ==========")
print(validation.to_string(index=False))

print("\n========== 전처리 요약 ==========")
print(summary.to_string(index=False))

if not validation["통과 여부"].all():
    raise ValueError("전처리 검증에 실패한 항목이 있습니다.")

print("\n모든 전처리 검증 통과")
