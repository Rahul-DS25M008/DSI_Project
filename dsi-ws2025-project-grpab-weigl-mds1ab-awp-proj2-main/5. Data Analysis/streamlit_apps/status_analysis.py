import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# =========================
# 1. DATA LOADING
# =========================

@st.cache_data
def load_data():
    """
    Load the base company table and compute lifecycle variables.
    Replace the CSV part with your DB query if you prefer.
    """

    from dotenv import load_dotenv
    import psycopg2

    load_dotenv()
    conn = psycopg2.connect(
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname"),
        user=os.getenv("user"),
        password=os.getenv("password"),
    )
    query = """
    SELECT
      c.id          AS company_id,
      c.company_name,
      c.start_year,
      c.funding_amount,
      s.is_active,
      s.status,
      ci.city_name,
      co.country_code,
      co.country
    FROM company c
    LEFT JOIN status s ON c.status_id = s.status_id
    LEFT JOIN city ci   ON c.city_id = ci.city_id
    LEFT JOIN country co ON ci.country_code = co.country_code;
    """
    df = pd.read_sql(query, conn)
    conn.close()

    CURRENT_YEAR = datetime.now().year

    df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")
    df["start_year"] = df["start_year"].apply(
    lambda y: y if pd.notna(y) and 1990 <= y <= 2025 else None)

    df["age"] = CURRENT_YEAR - df["start_year"]

    def map_lifecycle(row):
        if row["status"] == "active" or row["is_active"] is True:
            return "active"
        elif row["status"] == "inactive" or row["is_active"] is False:
            return "inactive"
        else:
            return "unknown"

    df["lifecycle_status"] = df.apply(map_lifecycle, axis=1)

    # lifecycle summary (for the status vs funding table)
    lifecycle_summary = (
        df[df["age"].notna()]
        .groupby("lifecycle_status")
        .agg(
            n_companies=("company_id", "count"),
            median_age=("age", "median"),
            mean_age=("age", "mean"),
            n_with_funding=("funding_amount", lambda x: x.notna().sum()),
            median_funding=("funding_amount", "median"),
            mean_funding=("funding_amount", "mean"),
        )
        .reset_index()
    )

    # country-level ratios
    country_status = (
        df.groupby(["country", "lifecycle_status"])["company_id"]
          .count()
          .reset_index(name="n_companies")
    )

    total_per_country = (
        country_status
        .groupby("country")["n_companies"]
        .sum()
        .reset_index(name="total_companies")
    )

    country_status = country_status.merge(total_per_country, on="country")
    country_status["ratio"] = (
        country_status["n_companies"] / country_status["total_companies"]
    )

    country_pivot = country_status.pivot_table(
        index="country",
        columns="lifecycle_status",
        values="ratio",
        fill_value=0,
    ).reset_index()

    counts_per_country = (
        df.groupby("country")["company_id"]
          .count()
          .reset_index(name="n_companies")
    )
    country_pivot = country_pivot.merge(counts_per_country, on="country")
    country_pivot = country_pivot[country_pivot["n_companies"] >= 200]

    # Making sure columns exist
    for col in ["active", "inactive", "unknown"]:
        if col not in country_pivot.columns:
            country_pivot[col] = 0.0

    return df, lifecycle_summary, country_pivot


# =========================
# 2. STREAMLIT LAYOUT
# =========================

st.set_page_config(
    page_title="Startup Lifecycle & Survival",
    layout="wide"
)

st.title("Startup Lifecycle & Survival Analysis")
st.markdown(
    "This dashboard explores **company age, status, funding, and country-level survival patterns** "
    "based on the unified startup dataset."
)

df, lifecycle_summary, country_pivot = load_data()

if df.empty:
    st.warning("Dataframe is empty. Plug in your real data in `load_data()`.")
    st.stop()

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters")

countries = sorted(df["country"].dropna().unique())
selected_countries = st.sidebar.multiselect(
    "Countries",
    options=countries,
    default=countries
)

status_options = ["active", "inactive", "unknown"]
selected_status = st.sidebar.multiselect(
    "Lifecycle status",
    options=status_options,
    default=status_options
)

min_age = int(df["age"].min())
max_age = int(df["age"].max())
age_range = st.sidebar.slider(
    "Age range (years)",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

mask = (
    df["country"].isin(selected_countries)
    & df["lifecycle_status"].isin(selected_status)
    & df["age"].between(age_range[0], age_range[1])
)
df_filtered = df[mask].copy()

# =========================
# 3. KPI SECTION
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Startups in selection", f"{len(df_filtered):,}")

with col2:
    active_share = (
        (df_filtered["lifecycle_status"] == "active").mean()
        if len(df_filtered) > 0 else 0
    )
    st.metric("Active share", f"{active_share:.1%}")

with col3:
    med_age = df_filtered["age"].median()
    st.metric("Median age (years)", f"{med_age:.1f}")

with col4:
    funding_cov = df_filtered["funding_amount"].notna().mean()
    st.metric("Funding coverage", f"{funding_cov:.1%}")

st.markdown("---")

# =========================
# 4. STATUS DISTRIBUTION
# =========================
st.subheader("Lifecycle Status Breakdown")

status_counts = (
    df_filtered["lifecycle_status"]
    .value_counts()
    .reindex(status_options)
    .fillna(0)
    .reset_index()
)
status_counts.columns = ["lifecycle_status", "n"]

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(
    data=status_counts,
    x="lifecycle_status",
    y="n",
    palette=["#4c9a8a", "#e0826b", "#c4b7a6"],
    ax=ax
)
ax.set_xlabel("Lifecycle Status")
ax.set_ylabel("Number of Companies")
ax.set_title("Distribution of Lifecycle Status (filtered selection)")
st.pyplot(fig)

# =========================
# 5. AGE DISTRIBUTION
# =========================
st.subheader("Age Distribution of Companies")

fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(
    data=df_filtered,
    x="age",
    bins=40,
    kde=True,
    ax=ax,
    color="#4c9a8a"
)
ax.set_xlabel("Age (years)")
ax.set_ylabel("Number of Companies")
ax.set_title("Distribution of Company Age")
st.pyplot(fig)

# =========================
# 6. AGE BY LIFECYCLE STATUS
# =========================
st.subheader("Company Age by Lifecycle Status")

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(
    data=df_filtered,
    x="lifecycle_status",
    y="age",
    order=status_options,
    palette=["#4c9a8a", "#e0826b", "#c4b7a6"],
    ax=ax
)
ax.set_xlabel("Lifecycle Status")
ax.set_ylabel("Age (years)")
ax.set_title("Age Distribution by Lifecycle Status")
st.pyplot(fig)

# =========================
# 7. FUNDING VS LIFECYCLE
# =========================
st.subheader("Funding & Survival")

st.markdown(
    "This table summarises how funding relates to lifecycle status "
    "(computed over the **full dataset**, not just the filtered slice)."
)
st.dataframe(
    lifecycle_summary.style.format(
        {
            "n_companies": "{:,.0f}",
            "median_age": "{:.1f}",
            "mean_age": "{:.1f}",
            "n_with_funding": "{:,.0f}",
            "median_funding": "{:,.0f}",
            "mean_funding": "{:,.0f}",
        }
    ),
    use_container_width=True,
)

# Optional scatter: funding vs age in selection
df_funded = df_filtered[df_filtered["funding_amount"].notna()]

if not df_funded.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=df_funded,
        x="age",
        y="funding_amount",
        hue="lifecycle_status",
        hue_order=status_options,
        alpha=0.6,
        ax=ax
    )
    ax.set_yscale("log")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Funding (log scale)")
    ax.set_title("Funding vs Age by Lifecycle Status (filtered selection)")
    st.pyplot(fig)
else:
    st.info("No funding data in the current selection for the scatter plot.")

# =========================
# 8. COUNTRY-LEVEL SURVIVAL
# =========================
st.subheader("Active vs Inactive Ratios by Country")

min_companies_country = st.slider(
    "Minimum number of companies per country (for inclusion)",
    min_value=50,
    max_value=int(country_pivot["n_companies"].max()),
    value=200,
    step=50,
)

country_plot_df = country_pivot[
    country_pivot["n_companies"] >= min_companies_country
].copy()

country_plot_df = country_plot_df.sort_values("active", ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(
    country_plot_df["country"],
    country_plot_df["active"],
    label="Active ratio",
    color="#4c9a8a",
)
ax.bar(
    country_plot_df["country"],
    country_plot_df["inactive"],
    bottom=country_plot_df["active"],
    label="Inactive ratio",
    color="#e0826b",
)

ax.set_ylim(0, 1)
ax.set_ylabel("Ratio")
ax.set_xlabel("Country")
ax.set_title("Active vs Inactive Startup Ratios by Country")
ax.legend()
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
st.pyplot(fig)
