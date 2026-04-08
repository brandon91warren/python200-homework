import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prefect import flow, get_run_logger, task
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "assignments" / "resources" / "happiness_project"
OUTPUT_DIR = BASE_DIR / "outputs"


def normalize_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def detect_csv_settings(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as f:
        lines = [next(f) for _ in range(5)]

    first_data_line = lines[1] if len(lines) > 1 else ""
    semicolon_count = first_data_line.count(";")
    comma_count = first_data_line.count(",")

    if semicolon_count > comma_count:
        sep = ";"
        decimal = ","
    else:
        sep = ","
        decimal = "."

    return sep, decimal


def extract_year(file_path):
    match = re.search(r"(20\d{2})", file_path.name)
    if not match:
        raise ValueError(f"Could not extract year from filename: {file_path.name}")
    return int(match.group(1))


def standardize_columns(df):
    df = df.copy()
    df.columns = [normalize_name(col) for col in df.columns]

    rename_map = {
        "country_name": "country",
        "country_or_region": "country",
        "country_region": "country",
        "regional_indicator": "region",
        "score": "happiness_score",
        "ladder_score": "happiness_score",
        "economy_gdp_per_capita": "gdp_per_capita",
        "logged_gdp_per_capita": "gdp_per_capita",
        "economy": "gdp_per_capita",
        "health_healthy_life_expectancy": "healthy_life_expectancy",
        "health_life_expectancy": "healthy_life_expectancy",
        "family": "social_support",
        "freedom_to_make_life_choices": "freedom",
        "trust_government_corruption": "corruption",
        "perceptions_of_corruption": "corruption",
        "trust_corruption": "corruption",
        "overall_rank": "ranking",
    }

    df = df.rename(columns={col: rename_map.get(col, col) for col in df.columns})
    return df


def coerce_numeric_columns(df):
    df = df.copy()
    protected_cols = {"country", "region", "year"}

    for col in df.columns:
        if col not in protected_cols:
            series = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .replace({"nan": np.nan, "": np.nan})
            )
            try:
                df[col] = pd.to_numeric(series)
            except (ValueError, TypeError):
                df[col] = series

    return df


@task(retries=3, retry_delay_seconds=2)
def load_multiple_years():
    logger = get_run_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_paths = sorted(DATA_DIR.glob("world_happiness_*.csv"))
    if not file_paths:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    frames = []

    for file_path in file_paths:
        year = extract_year(file_path)
        sep, decimal = detect_csv_settings(file_path)
        logger.info(f"Loading {file_path.name} with sep='{sep}' and decimal='{decimal}'")
        df = pd.read_csv(file_path, sep=sep, decimal=decimal, encoding="utf-8-sig")
        df = standardize_columns(df)
        df["year"] = year
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = coerce_numeric_columns(merged)

    merged_path = OUTPUT_DIR / "merged_happiness.csv"
    merged.to_csv(merged_path, index=False)

    logger.info(f"Merged dataset saved to {merged_path}")
    logger.info(f"Merged shape: {merged.shape}")

    return merged


@task
def descriptive_statistics(df):
    logger = get_run_logger()

    overall_mean = df["happiness_score"].mean()
    overall_median = df["happiness_score"].median()
    overall_std = df["happiness_score"].std()

    logger.info(f"Overall happiness mean: {overall_mean:.4f}")
    logger.info(f"Overall happiness median: {overall_median:.4f}")
    logger.info(f"Overall happiness standard deviation: {overall_std:.4f}")

    by_year = df.groupby("year")["happiness_score"].mean().sort_index()
    logger.info("Mean happiness score by year:")
    for year, value in by_year.items():
        logger.info(f"{year}: {value:.4f}")

    if "region" in df.columns:
        by_region = df.groupby("region")["happiness_score"].mean().sort_values(ascending=False)
        logger.info("Mean happiness score by region:")
        for region, value in by_region.items():
            logger.info(f"{region}: {value:.4f}")
    else:
        by_region = pd.Series(dtype=float)
        logger.info("Region column not found; skipping regional descriptive statistics.")

    return {
        "overall_mean": overall_mean,
        "overall_median": overall_median,
        "overall_std": overall_std,
        "by_year": by_year.to_dict(),
        "by_region": by_region.to_dict(),
    }


@task
def visual_exploration(df):
    logger = get_run_logger()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(df["happiness_score"].dropna(), bins=20)
    plt.title("Distribution of Happiness Scores")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    hist_path = OUTPUT_DIR / "happiness_histogram.png"
    plt.savefig(hist_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved histogram to {hist_path}")

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="year", y="happiness_score")
    plt.title("Happiness Score by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    box_path = OUTPUT_DIR / "happiness_by_year.png"
    plt.savefig(box_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved boxplot to {box_path}")

    if "gdp_per_capita" in df.columns:
        plot_df = df[["gdp_per_capita", "happiness_score"]].dropna()
        plt.figure(figsize=(10, 6))
        plt.scatter(plot_df["gdp_per_capita"], plot_df["happiness_score"])
        plt.title("GDP per Capita vs Happiness Score")
        plt.xlabel("GDP per Capita")
        plt.ylabel("Happiness Score")
        scatter_path = OUTPUT_DIR / "gdp_vs_happiness.png"
        plt.savefig(scatter_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved scatter plot to {scatter_path}")
    else:
        logger.info("GDP per capita column not found; skipping scatter plot.")

    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    heatmap_path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved heatmap to {heatmap_path}")


@task
def hypothesis_testing(df):
    logger = get_run_logger()

    scores_2019 = df.loc[df["year"] == 2019, "happiness_score"].dropna()
    scores_2020 = df.loc[df["year"] == 2020, "happiness_score"].dropna()

    t_stat, p_value = stats.ttest_ind(scores_2019, scores_2020, equal_var=False)

    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()

    logger.info(f"2019 vs 2020 t-statistic: {t_stat:.4f}")
    logger.info(f"2019 vs 2020 p-value: {p_value:.6f}")
    logger.info(f"Mean happiness in 2019: {mean_2019:.4f}")
    logger.info(f"Mean happiness in 2020: {mean_2020:.4f}")

    if p_value < 0.05:
        if mean_2020 < mean_2019:
            interpretation = "There is a statistically significant drop in happiness from 2019 to 2020, suggesting the pandemic period may have been associated with lower global happiness."
        else:
            interpretation = "There is a statistically significant increase in happiness from 2019 to 2020, suggesting a measurable shift in global happiness between these years."
    else:
        interpretation = "The difference in happiness between 2019 and 2020 is not statistically significant at alpha = 0.05, so this dataset does not provide strong evidence of a real change rather than random variation."

    logger.info(interpretation)

    second_test_result = None

    if "region" in df.columns:
        regional_means = df.groupby("region")["happiness_score"].mean().sort_values(ascending=False)

        if len(regional_means) >= 2:
            high_region = regional_means.index[0]
            low_region = regional_means.index[-1]

            group_high = df.loc[df["region"] == high_region, "happiness_score"].dropna()
            group_low = df.loc[df["region"] == low_region, "happiness_score"].dropna()

            t2, p2 = stats.ttest_ind(group_high, group_low, equal_var=False)

            logger.info(f"Second test comparing regions: {high_region} vs {low_region}")
            logger.info(f"Second test t-statistic: {t2:.4f}")
            logger.info(f"Second test p-value: {p2:.6f}")
            logger.info(f"{high_region} mean happiness: {group_high.mean():.4f}")
            logger.info(f"{low_region} mean happiness: {group_low.mean():.4f}")

            second_test_result = {
                "regions": (high_region, low_region),
                "t_stat": t2,
                "p_value": p2,
            }

    return {
        "prepost_t_stat": t_stat,
        "prepost_p_value": p_value,
        "mean_2019": mean_2019,
        "mean_2020": mean_2020,
        "interpretation": interpretation,
        "second_test": second_test_result,
    }


@task
def correlation_and_multiple_comparisons(df):
    logger = get_run_logger()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in {"happiness_score", "year", "ranking"}]

    results = []

    for col in numeric_cols:
        temp = df[["happiness_score", col]].dropna()
        if len(temp) < 3:
            continue

        r, p = stats.pearsonr(temp[col], temp["happiness_score"])
        results.append({"variable": col, "correlation": r, "p_value": p})

    num_tests = len(results)
    adjusted_alpha = 0.05 / num_tests if num_tests > 0 else None

    logger.info(f"Number of correlation tests: {num_tests}")
    if adjusted_alpha is not None:
        logger.info(f"Bonferroni-adjusted alpha: {adjusted_alpha:.6f}")

    for result in results:
        sig_005 = result["p_value"] < 0.05
        sig_bonf = result["p_value"] < adjusted_alpha if adjusted_alpha is not None else False
        logger.info(
            f"{result['variable']}: r={result['correlation']:.4f}, "
            f"p={result['p_value']:.6f}, "
            f"significant_at_0_05={sig_005}, "
            f"significant_after_bonferroni={sig_bonf}"
        )

    strongest = None
    bonferroni_significant = [r for r in results if adjusted_alpha is not None and r["p_value"] < adjusted_alpha]

    if bonferroni_significant:
        strongest = max(bonferroni_significant, key=lambda x: abs(x["correlation"]))
    elif results:
        strongest = max(results, key=lambda x: abs(x["correlation"]))

    return {
        "results": results,
        "num_tests": num_tests,
        "adjusted_alpha": adjusted_alpha,
        "strongest": strongest,
    }


@task
def summary_report(df, hypothesis_results, correlation_results):
    logger = get_run_logger()

    total_countries = df["country"].nunique() if "country" in df.columns else df.shape[0]
    total_years = df["year"].nunique()

    logger.info(f"Total number of countries in merged dataset: {total_countries}")
    logger.info(f"Total number of years in merged dataset: {total_years}")

    if "region" in df.columns:
        regional_means = df.groupby("region")["happiness_score"].mean().sort_values(ascending=False)
        top_3 = regional_means.head(3)
        bottom_3 = regional_means.tail(3)

        logger.info(
            "Top 3 regions by mean happiness score: "
            + ", ".join([f"{region} ({value:.3f})" for region, value in top_3.items()])
        )
        logger.info(
            "Bottom 3 regions by mean happiness score: "
            + ", ".join([f"{region} ({value:.3f})" for region, value in bottom_3.items()])
        )

    logger.info(f"Pre/post-2020 t-test result: {hypothesis_results['interpretation']}")

    strongest = correlation_results.get("strongest")
    if strongest:
        logger.info(
            f"Most strongly correlated variable with happiness score (after Bonferroni correction when available): "
            f"{strongest['variable']} with r={strongest['correlation']:.4f} and p={strongest['p_value']:.6f}"
        )
    else:
        logger.info("No valid correlation results were available.")


@flow
def happiness_pipeline():
    df = load_multiple_years()
    descriptive_statistics(df)
    visual_exploration(df)
    hypothesis_results = hypothesis_testing(df)
    correlation_results = correlation_and_multiple_comparisons(df)
    summary_report(df, hypothesis_results, correlation_results)


if __name__ == "__main__":
    happiness_pipeline()