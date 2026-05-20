import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool


api_key = os.getenv("OPENAI_API_KEY")

DATA_PATH = Path("../assignments_01/outputs/merged_happiness.csv")
FALLBACK_DIR = Path("happiness_project")
OUTPUT_DIR = Path("outputs")

df = None


@tool
def load_happiness_data() -> dict:
    """
    Load the World Happiness dataset into memory.

    Returns:
        A dictionary containing the dataset shape and column names.
    """
    global df

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        files = sorted(FALLBACK_DIR.glob("*.csv"))

        if not files:
            return {
                "error": f"No merged file found at {DATA_PATH} and no CSV files found in {FALLBACK_DIR}."
            }

        frames = []

        for file in files:
            year_text = "".join(ch for ch in file.stem if ch.isdigit())
            year = int(year_text) if year_text else None

            temp_df = pd.read_csv(file)
            temp_df.columns = (
                temp_df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace(".", "", regex=False)
            )

            rename_map = {
                "country_or_region": "country",
                "country": "country",
                "happiness_score": "happiness_score",
                "score": "happiness_score",
                "economy_gdp_per_capita": "gdp_per_capita",
                "gdp_per_capita": "gdp_per_capita",
                "regional_indicator": "region",
                "region": "region"
            }

            temp_df = temp_df.rename(columns=rename_map)

            if "year" not in temp_df.columns:
                temp_df["year"] = year

            frames.append(temp_df)

        df = pd.concat(frames, ignore_index=True)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
    )

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "data": df.to_dict(orient="records")
    }

@tool
def summarize_column(column: str) -> dict:
    """
    Return descriptive statistics for a single column in the loaded dataset.

    Args:
        column: The column name to summarize.

    Returns:
        A dictionary containing descriptive statistics for the requested column.
    """
    if df is None:
        return {"error": "No data is loaded."}

    if column not in df.columns:
        return {"error": f"Column not found: {column}"}

    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: The first numeric column.
        col2: The second numeric column.

    Returns:
        A dictionary containing the column names, Pearson correlation coefficient, and p-value.
    """
    if df is None:
        return {"error": "No data is loaded."}

    if col1 not in df.columns:
        return {"error": f"Column not found: {col1}"}

    if col2 not in df.columns:
        return {"error": f"Column not found: {col2}"}

    clean_df = df[[col1, col2]].dropna()

    try:
        r, p = pearsonr(clean_df[col1], clean_df[col2])
        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 4)
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """
    Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The column name to rank countries by.
        year: The year to filter the dataset by.
        n: The number of countries to return.

    Returns:
        A dictionary containing the top countries and their values for the requested column.
    """
    if df is None:
        return {"error": "No data is loaded."}

    if "year" not in df.columns:
        return {"error": "The dataset does not contain a year column."}

    if "country" not in df.columns:
        return {"error": "The dataset does not contain a country column."}

    if column not in df.columns:
        return {"error": f"Column not found: {column}"}

    year_df = df[df["year"] == year]

    if year_df.empty:
        return {"error": f"No rows found for year {year}."}

    top_rows = (
        year_df
        .sort_values(by=column, ascending=False)
        .head(n)[["country", column]]
    )

    return {
        "year": year,
        "column": column,
        "top_countries": top_rows.to_dict(orient="records")
    }


SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""


def build_agent():
    model = OpenAIServerModel(
        api_key=api_key,
        model_id="gpt-4o-mini"
    )

    return CodeAgent(
        tools=[
            load_happiness_data,
            summarize_column,
            compute_correlation,
            get_top_n_countries
        ],
        model=model,
        instructions=SYSTEM_PROMPT,
        additional_authorized_imports=[
            "pandas",
            "matplotlib",
            "matplotlib.pyplot",
            "scipy.stats",
            "os",
            "pathlib"
        ],
        max_steps=8
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    agent = build_agent()

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    my_query_1 = "Which region had the highest average happiness_score overall? Use code if needed."
    print(f"\n--- My Query 1: {my_query_1} ---")
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: This should trigger code generation because there is no specific tool for grouping by region and averaging.

    my_query_2 = "Show me the top 3 countries for gdp_per_capita in 2020."
    print(f"\n--- My Query 2: {my_query_2} ---")
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: This should trigger tool use because get_top_n_countries directly handles ranking countries by a column for a year. 
    


# --- Reflection ---
#
# 1. In Query 3, the agent communicated statistical significance by using the p-value
#    returned from the compute_correlation tool. It used the p-value correctly if it
#    compared it to a common threshold like 0.05. If p < 0.05, the relationship is
#    usually described as statistically significant.
#
# 2. One thing that surprised me was how the agent could decide when to use a tool
#    versus when to write its own Python code. For example, the regional line chart
#    required custom plotting code because none of the tools directly created that
#    type of chart.
#
# 3. One additional useful tool would be a plot_by_group tool. It would take a numeric
#    column, a grouping column, and a time column, then save a line chart automatically.
#    This would help answer questions like how happiness_score changed over time by
#    region without needing the CodeAgent to write custom plotting code each time.
