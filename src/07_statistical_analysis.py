
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal

# ==========================================
# 1. 데이터 불러오기
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data/processed/fire_industrial_clean.parquet"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT_PATH)

print("========== 통계 분석 ==========")
print(f"전체 산업시설 화재: {len(df):,}건")

results = []

# ==========================================
# 2. 통계 분석 함수
# ==========================================

def chi_square_test(data, category_col, outcome_col, test_name):
    subset = data[[category_col, outcome_col]].dropna()
    table = pd.crosstab(
        subset[category_col],
        subset[outcome_col]
    )

    if table.shape[0] < 2 or table.shape[1] < 2:
        print(f"{test_name}: 검정 불가 (범주 부족)")
        return

    chi2, p, dof, expected = chi2_contingency(
        table, correction=False
    )

    n = table.to_numpy().sum()
    min_dim = min(table.shape) - 1

    cramers_v = np.sqrt(
        chi2 / (n * min_dim)
    )

    low_expected = (expected < 5).sum()
    low_expected_pct = (
            low_expected / expected.size * 100
    )

    print(f"\n[{test_name}]")
    print(f"표본 수: {n:,}")
    print(f"카이제곱 통계량: {chi2:.4f}")
    print(f"p-value: {p:.6g}")
    print(f"Cramér's V: {cramers_v:.4f}")
    print(f"기대빈도 5 미만 셀 비율: {low_expected_pct:.2f}%")

    results.append({
        "검정명": test_name,
        "검정방법": "Chi-square",
        "표본수": n,
        "검정통계량": chi2,
        "p값": p,
        "효과크기": cramers_v,
        "효과크기종류": "Cramer's V",
        "검정가정주의": low_expected_pct > 20
    })


# ==========================================
# 3. 인명피해 여부 생성
# ==========================================

df["인명피해발생"] = np.where(
    df["인명피해(명)"].isna(),
    pd.NA,
    df["인명피해(명)"] > 0
)

# ==========================================
# 4. 발화요인과 인명피해 검정
# ==========================================

# 표본이 적은 범주는 별도 묶음 처리
cause_counts = df["발화요인"].value_counts()

df["발화요인_분석"] = df["발화요인"].where(
    df["발화요인"].map(cause_counts) >= 100,
    "소수범주"
)

chi_square_test(
    df,
    "발화요인_분석",
    "인명피해발생",
    "발화요인과 인명피해 발생"
)

# ==========================================
# 5. 발화요인별 재산피해 검정
# ==========================================

damage_col = "재산피해(천원)"

damage_df = df[
    df[damage_col].notna()
    & (df[damage_col] >= 0)
    ].copy()

groups = [
    group[damage_col].to_numpy()
    for _, group in damage_df.groupby("발화요인_분석")
    if len(group) >= 20
]

if len(groups) >= 2:
    h_stat, p_value = kruskal(*groups)

    n = sum(len(group) for group in groups)
    k = len(groups)

    epsilon_sq = max(
        0.0,
        (h_stat - k + 1) / (n - k)
    )

    print("\n[발화요인별 재산피해 분포]")
    print(f"표본 수: {n:,}")
    print(f"검정 집단 수: {k}")
    print(f"Kruskal-Wallis H: {h_stat:.4f}")
    print(f"p-value: {p_value:.6g}")
    print(f"Epsilon squared: {epsilon_sq:.4f}")

    results.append({
        "검정명": "발화요인별 재산피해 분포",
        "검정방법": "Kruskal-Wallis",
        "표본수": n,
        "검정통계량": h_stat,
        "p값": p_value,
        "효과크기": epsilon_sq,
        "효과크기종류": "Epsilon squared",
        "검정가정주의": False
    })

# ==========================================
# 6. 발화층과 인명피해 검정
# ==========================================

# 괄호 표기가 지하층이라는 해석은 잠정적임
floor_df = df[
    df["발화층_구분"].isin(["지상", "지하"])
].copy()

chi_square_test(
    floor_df,
    "발화층_구분",
    "인명피해발생",
    "발화층과 인명피해 발생 (잠정)"
)

# ==========================================
# 7. 결과 저장
# ==========================================

result_df = pd.DataFrame(results)

OUTPUT_PATH = REPORT_DIR / "statistical_test_results.csv"

result_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== 통계 검정 완료 ==========")
print(result_df.to_string(index=False))
print(f"\n저장 위치: {OUTPUT_PATH}")

