import os
import ast

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

sns.set(style="whitegrid")
load_dotenv()

# =========================================================
# DB CONNECTION & DATA LOADING
# =========================================================

@st.cache_data
def get_connection_params():
    return dict(
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname"),
        user=os.getenv("user"),
        password=os.getenv("password"),
    )

@st.cache_data
def load_industry_base():
    """Recreating df_industry_base from the notebook using the DB."""
    params = get_connection_params()
    conn = psycopg2.connect(**params)

    query = """
    SELECT
        c.id                AS company_id,
        c.company_name,
        c.funding_amount,
        c.start_year,
        c.website,
        ci.city_name,
        co.country_code,
        co.country,
        s.is_active,
        s.status,
        ARRAY_AGG(DISTINCT it.industry_tag ORDER BY it.industry_tag) AS industries
    FROM company c
    LEFT JOIN city ci
        ON c.city_id = ci.city_id
    LEFT JOIN country co
        ON ci.country_code = co.country_code
    LEFT JOIN status s
        ON c.status_id = s.status_id
    LEFT JOIN company_industries ci_map
        ON c.id = ci_map.company_id
    LEFT JOIN industry_tags it
        ON ci_map.industry_id = it.industry_id
    GROUP BY
        c.id,
        c.company_name,
        c.funding_amount,
        c.start_year,
        c.website,
        ci.city_name,
        co.country_code,
        co.country,
        s.is_active,
        s.status
    ORDER BY c.id;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # ensuring that the industries is a list
    df["industries"] = df["industries"].apply(
        lambda x: [i for i in x if i is not None] if isinstance(x, (list, tuple)) else []
    )

    # feature engineering (same as notebook)
    from datetime import datetime
    CURRENT_YEAR = datetime.now().year

    df["company_age"] = df["start_year"].apply(
        lambda y: CURRENT_YEAR - int(y) if pd.notna(y) else np.nan
    )
    df["has_funding"] = df["funding_amount"].notna()
    df["log_funding"] = df["funding_amount"].apply(
        lambda x: np.log1p(x) if pd.notna(x) and x > 0 else np.nan
    )
    df["n_industries"] = df["industries"].apply(lambda lst: len(lst) if isinstance(lst, list) else 0)

    return df


@st.cache_data
def build_industry_summary(df_industry_base: pd.DataFrame):
    """Rebuilding industry_summary & filtered version used in notebook"""
    df_industry = df_industry_base.copy()
    df_industry_long = df_industry.explode("industries").copy()
    df_industry_long = df_industry_long[
        df_industry_long["industries"].notna()
        & (df_industry_long["industries"] != "")
    ]

    industry_counts = (
        df_industry_long
        .groupby("industries")["company_id"]
        .nunique()
        .reset_index(name="n_companies")
    )

    industry_funding_presence = (
        df_industry_long
        .assign(has_funding=df_industry_long["funding_amount"].notna())
        .groupby(["industries", "has_funding"])["company_id"]
        .nunique()
        .unstack(fill_value=0)
        .rename(columns={False: "n_no_funding", True: "n_with_funding"})
        .reset_index()
    )

    # funded subset for funding stats
    df_funded = df_industry_base.dropna(subset=["funding_amount"])
    df_funded_long = df_funded.explode("industries").copy()
    df_funded_long = df_funded_long[
        df_funded_long["industries"].notna()
        & (df_funded_long["industries"] != "")
    ]

    industry_funding_stats = (
        df_funded_long
        .groupby("industries")["funding_amount"]
        .agg(
            n_funded_companies="count",
            median_funding="median",
            mean_funding="mean",
            max_funding="max"
        )
        .reset_index()
    )

    industry_summary = (
        industry_counts
        .merge(industry_funding_presence, on="industries", how="left")
        .merge(industry_funding_stats, on="industries", how="left")
    )

    # filtered version for bubble / funding charts
    ind = industry_summary.copy()
    for col in ["n_funded_companies", "median_funding", "mean_funding"]:
        if col in ind.columns:
            ind[col] = ind[col].astype(float)

    ind["funding_ratio"] = ind["n_with_funding"] / ind["n_companies"]

    min_companies = 20
    min_funded = 5
    max_median = 1e9

    ind_filtered = ind[
        (ind["n_companies"] >= min_companies) &
        (ind["n_funded_companies"] >= min_funded) &
        (ind["median_funding"].notna()) &
        (ind["median_funding"] < max_median)
    ].copy()

    return industry_summary, ind_filtered


@st.cache_data
def load_nlp_clusters(csv_path: str):
    """
    Load df_nlp from Part A (with precomputed clusters).
    Expects columns like: name, description, clean_text, cluster,
    industries_list, funding_usd, started_in, ...
    """
    df = pd.read_csv(csv_path)

    if "industries_list" in df.columns and df["industries_list"].dtype == object:
        def parse_list(x):
            try:
                return ast.literal_eval(x)
            except Exception:
                return []
        df["industries_list"] = df["industries_list"].apply(parse_list)

    # basic cleaning
    if "started_in" in df.columns:
        df["started_in"] = pd.to_numeric(df["started_in"], errors="coerce")

    if "funding_usd" in df.columns:
        df["funding_usd"] = pd.to_numeric(df["funding_usd"], errors="coerce")

    return df


@st.cache_data
def build_cluster_summaries(df_nlp: pd.DataFrame):
    """Rebuilding topic_df, df_ci, cluster_industry_counts, cluster_funding, cluster_summary from notebook"""
    # topic_df: simple TF-IDF keyword extraction per cluster (reuse minimal version)
    from sklearn.feature_extraction.text import TfidfVectorizer

    df_clusters = df_nlp[df_nlp["cluster"] != -1].copy()
    clusters = sorted(df_clusters["cluster"].unique())

    topics = []
    for c in clusters:
        texts = df_clusters[df_clusters["cluster"] == c]["clean_text"].astype(str).tolist()
        if len(texts) < 3:
            continue
        vec = TfidfVectorizer(max_features=20)
        tfidf = vec.fit_transform(texts)
        keywords = vec.get_feature_names_out()
        topics.append({
            "cluster": c,
            "size": len(texts),
            "keywords": ", ".join(keywords[:7])
        })

    topic_df = pd.DataFrame(topics).sort_values("size", ascending=False)

    df_ci = df_clusters.explode("industries_list").copy()
    df_ci = df_ci[df_ci["industries_list"].notna()]
    df_ci = df_ci[df_ci["industries_list"].astype(str).str.strip() != ""]

    cluster_industry_counts = (
        df_ci
        .groupby(["cluster", "industries_list"])
        .size()
        .reset_index(name="count")
    )

    cluster_funding = (
        df_ci[df_ci["funding_usd"].notna()]
        .groupby("cluster")["funding_usd"]
        .agg(
            n_funded="count",
            median_funding="median",
            mean_funding="mean",
            max_funding="max"
        )
        .reset_index()
    )

    top_industry_per_cluster = (
        cluster_industry_counts
        .sort_values(["cluster", "count"], ascending=[True, False])
        .groupby("cluster")
        .head(1)
        .rename(columns={"industries_list": "top_industry", "count": "top_industry_count"})
    )

    cluster_summary = (
        topic_df
        .merge(cluster_funding, on="cluster", how="left")
        .merge(top_industry_per_cluster, on="cluster", how="left")
        .sort_values("size", ascending=False)
    )

    return topic_df, df_ci, cluster_industry_counts, cluster_funding, cluster_summary


# =========================================================
# STREAMLIT LAYOUT
# =========================================================

st.set_page_config(
    page_title="EU Companies – Industries & Themes",
    layout="wide"
)

st.title("Industry & Theme Analysis of EU Startups")

# Sidebar controls
st.sidebar.header("Data & Filters")

st.sidebar.markdown("**NLP cluster CSV (from notebook)**")
nlp_csv_path = st.sidebar.text_input(
    "Path to `failory_nlp_clusters.csv`",
    value="/workspaces/dsi-ws2025-project-grpab-weigl-mds1ab-awp-proj2/5. Data Analysis/data/failory_nlp_clusters.csv"
)

# Loading the data
df_industry_base = load_industry_base()
industry_summary, ind_filtered = build_industry_summary(df_industry_base)

st.sidebar.markdown("---")

countries = sorted(df_industry_base["country"].dropna().unique())
selected_countries = st.sidebar.multiselect(
    "Filter by country (industry plots):",
    options=countries,
    default=countries
)

# Filtering industry data by selected countries
if selected_countries:
    df_industry_filtered = df_industry_base[df_industry_base["country"].isin(selected_countries)]
    industry_summary_filtered, ind_filtered_country = build_industry_summary(df_industry_filtered)
else:
    industry_summary_filtered, ind_filtered_country = industry_summary, ind_filtered

# NLP data (Failory clusters)
df_nlp = None
cluster_summary = None
df_ci = None
cluster_industry_counts = None

if os.path.exists(nlp_csv_path):
    df_nlp = load_nlp_clusters(nlp_csv_path)
    topic_df, df_ci, cluster_industry_counts, cluster_funding, cluster_summary = build_cluster_summaries(df_nlp)
else:
    st.sidebar.error("NLP CSV not found. Check the path.")


tab1, tab2 = st.tabs(["Industry Overview", "NLP Theme Clusters"])


# =========================================================
# TAB 1 — INDUSTRY OVERVIEW
# =========================================================
with tab1:
    st.subheader("Industry Overview")

    st.markdown(
        """
        This section answers two basic questions:
        1. **Where are most startups concentrated by industry?**  
        2. **Which industries attract more funding per startup?**
        """
    )

    ind_plot = industry_summary_filtered.copy()

    # Top industries by number of companies
    top_n = st.slider("Top N industries", 5, 40, 20, step=5)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top Industries by Company Count")
        top_size = (
            ind_plot.sort_values("n_companies", ascending=False)
                    .head(top_n)
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.barplot(
            data=top_size,
            y="industries",
            x="n_companies",
            ax=ax
        )
        ax.set_xlabel("Number of Companies")
        ax.set_ylabel("Industry")
        st.pyplot(fig)

    with col2:
        st.markdown("#### Top Industries by Median Funding (Filtered)")
        top_median = (
            ind_filtered_country
            .sort_values("median_funding", ascending=False)
            .head(top_n)
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.barplot(
            data=top_median,
            y="industries",
            x="median_funding",
            ax=ax
        )
        ax.set_xlabel("Median Funding (€)")
        ax.set_ylabel("Industry")
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### Industry Size vs Median Funding (Bubble View)")

    bubble_df = ind_filtered_country.copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        bubble_df["n_companies"],
        bubble_df["median_funding"],
        s=np.clip(bubble_df["n_funded_companies"], 5, 300) * 2,
        c=bubble_df["funding_ratio"],
        cmap="viridis",
        alpha=0.7
    )
    ax.set_xlabel("Number of Companies")
    ax.set_ylabel("Median Funding (€)")
    ax.set_title("Industry Size vs Median Funding")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Share of Companies with Funding Data")
    st.pyplot(fig)

    st.markdown(
        """
        *Interpretation*:  
        - **Right side**: big industries.  
        - **High up**: industries with high typical funding.  
        - **Big bubbles**: industries with many funded companies.  
        - **Bright colors**: industries where funding data is well-covered.
        """
    )


# =========================================================
# TAB 2 — NLP THEME CLUSTERS
# =========================================================
with tab2:
    st.subheader("NLP-Derived Startup Themes")

    if df_nlp is None or cluster_summary is None:
        st.warning("NLP cluster data not loaded. Provide a valid CSV path in the sidebar.")
    else:
        st.markdown(
            """
            Here we look at **semantic clusters** of startups based on their descriptions:
            - Each cluster is a theme (e.g. FinTech infrastructure, biotech, cybersecurity).
            - We combine **keywords, industries, and funding** to interpret them.
            """
        )

        # Cluster selector
        cluster_ids = cluster_summary["cluster"].tolist()
        selected_cluster = st.selectbox(
            "Select a cluster to inspect:",
            options=cluster_ids,
            index=0
        )

        row = cluster_summary[cluster_summary["cluster"] == selected_cluster].iloc[0]

        st.markdown("### Cluster summary")
        st.write(
            {
                "Cluster": int(row["cluster"]),
                "Size (# startups)": int(row["size"]),
                "Keywords": row["keywords"],
                "Top industry": row["top_industry"],
                "Top industry count": int(row["top_industry_count"]) if not pd.isna(row["top_industry_count"]) else None,
                "Median funding": float(row["median_funding"]) if not pd.isna(row["median_funding"]) else None,
                "Mean funding": float(row["mean_funding"]) if not pd.isna(row["mean_funding"]) else None,
                "Max funding": float(row["max_funding"]) if not pd.isna(row["max_funding"]) else None,
            }
        )

        st.markdown("---")
        col1, col2 = st.columns([2, 1])

        # 1) Cluster growth over time
        with col1:
            st.markdown("#### Cluster Growth Over Time")

            cluster_year = (
                df_nlp[df_nlp["cluster"] != -1]
                .groupby(["cluster", "started_in"])
                .size()
                .reset_index(name="n")
            )

            top_clusters = cluster_summary["cluster"].head(10).tolist()
            selected_for_growth = st.multiselect(
                "Clusters to show in growth plot:",
                options=top_clusters,
                default=top_clusters[:5]
            )

            plot_df = cluster_year[
                cluster_year["cluster"].isin(selected_for_growth)
            ].copy()

            fig, ax = plt.subplots(figsize=(10, 5))
            if not plot_df.empty:
                sns.lineplot(
                    data=plot_df,
                    x="started_in",
                    y="n",
                    hue="cluster",
                    marker="o",
                    ax=ax
                )
                ax.set_xlabel("Start Year")
                ax.set_ylabel("Number of Startups")
                ax.set_title("Cluster Growth Over Time (Selected Clusters)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.info("No data for selected clusters / years.")

        # 2) Wordcloud for selected cluster
        with col2:
            st.markdown("#### Wordcloud for Selected Cluster")

            subset = df_nlp[df_nlp["cluster"] == selected_cluster]

            # Prefer clean_text if available, else description
            if "clean_text" in subset.columns:
                text_series = subset["clean_text"].dropna()
            else:
                text_series = subset["description"].astype(str).dropna()

            if not text_series.empty:
                big_text = " ".join(text_series.tolist())
                wc = WordCloud(
                    width=600,
                    height=400,
                    background_color="white",
                    max_words=100
                ).generate(big_text)

                fig_wc, ax_wc = plt.subplots(figsize=(5, 3))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("No text available for this cluster to generate a wordcloud.")

        st.markdown("---")

        # 3) Top clusters by median funding
        st.markdown("#### Top Clusters by Median Funding")

        top_k = st.slider("Show top K clusters", 3, 15, 10)

        top10_funding = (
            cluster_summary
            .dropna(subset=["median_funding"])
            .sort_values("median_funding", ascending=False)
            .head(top_k)
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=top10_funding,
            x="cluster",
            y="median_funding",
            palette="viridis",
            ax=ax
        )
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Median Funding (USD)")
        ax.set_title("Top Clusters by Median Funding")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)

        st.markdown("---")

        # 4) Optional: cluster × industry heatmap (static but nice)
        st.markdown("#### Industry Distribution Across Clusters (Top 25 Industries)")

        popular_inds = (
            cluster_industry_counts
            .groupby("industries_list")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(25)
            .index
        )

        pivot_ci = cluster_industry_counts[
            cluster_industry_counts["industries_list"].isin(popular_inds)
        ].pivot(
            index="cluster",
            columns="industries_list",
            values="count"
        ).fillna(0)

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            pivot_ci,
            cmap="viridis",
            linewidths=0.2,
            ax=ax
        )
        ax.set_title("Industry Distribution Across NLP Clusters (Top 25 Industries)")
        ax.set_xlabel("Industry")
        ax.set_ylabel("Cluster")
        st.pyplot(fig)
