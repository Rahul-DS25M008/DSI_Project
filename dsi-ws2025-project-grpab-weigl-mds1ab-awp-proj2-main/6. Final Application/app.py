# Library and Package Imports

import os
import ast
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import psycopg2
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Global configs and secrets loading.
sns.set(style="whitegrid")
load_dotenv()

st.set_page_config(
    page_title="European Startup Landscape Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shared helper functions for easier data handling across scripts
@st.cache_data
def get_connection_params():
    return dict(
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname"),
        user=os.getenv("user"),
        password=os.getenv("password"),
    )

@st.cache_resource
def get_conn():
    params = get_connection_params()
    return psycopg2.connect(**params)

def run_sql(query: str) -> pd.DataFrame:
    conn = get_conn()
    return pd.read_sql(query, conn)

def safe_title(title: str):
    st.markdown(f"# {title}")

# Page: Home (landing.py)
def render_home():
    st.markdown("""
    <style>

    /* ---------- Global Font & Base ---------- */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                    Roboto, Helvetica, Arial, sans-serif;
        color: #1f2933;
    }

    /* ---------- Page Container ---------- */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* ---------- Headings ---------- */
    h1 {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
    }

    h2 {
        font-size: 1.6rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    h3 {
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* ---------- Subtitle under Hero ---------- */
    .hero-subtitle {
        font-size: 1.15rem;
        color: #4b5563;
        margin-top: 0.3rem;
    }

    /* ---------- Fun Fact Boxes ---------- */
    .fun-fact-box {
        background-color: #f8fafc;
        padding: 20px 22px;
        border-radius: 12px;
        border-left: 5px solid #2563eb;
        margin-bottom: 20px;
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.06);
    }

    .fun-fact-box h3 {
        color: #2563eb;
        margin-bottom: 0.3rem;
    }

    .fun-fact-box p {
        font-size: 0.95rem;
        line-height: 1.5;
        color: #374151;
    }

    /* ---------- Horizontal Rule ---------- */
    hr {
        border: none;
        height: 1px;
        background-color: #e5e7eb;
        margin: 2rem 0;
    }
                
    /* ---------- Fun Fact Grid Cards: consistent height + layout ---------- */
    .fun-fact-box{
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 220px;           /* same height baseline */
    }

    /* Big “headline stat” */
    .fun-fact-stat{
        font-size: 2rem;
        font-weight: 800;
        color: #1d4ed8;
        line-height: 1.1;
        margin: 0.2rem 0 0.6rem 0;
    }

    .fun-fact-stat span{
        font-size: 0.95rem;
        font-weight: 600;
        color: #4b5563;
        margin-left: 0.4rem;
    }

    /* Bullet formatting inside cards */
    .fun-fact-bullets{
        margin: 0;
        padding-left: 1.1rem;
        color: #374151;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .fun-fact-bullets li{
        margin-bottom: 0.35rem;
    }

    /* Source line inside card (optional) */
    .fun-fact-footnote{
        margin-top: 0.8rem;
        font-size: 0.85rem;
        color: #6b7280;
    }
    </style>
    """, unsafe_allow_html=True)

    col_title, col_image = st.columns([2, 1])

    with col_title:
        st.markdown("# **The EU Startup Landscape Monitor**")
        st.markdown(
            "<div class='hero-subtitle'>Monitoring the pulse of innovation across the continent.</div>",
            unsafe_allow_html=True
        )

    with col_image:
        st.image("static/landing_page_1.jpg")

    st.markdown("---")
    st.markdown("## **Fascinating Facts from the Start-up Ecosystem** ")

    fact_cols = st.columns(3)

    with fact_cols[0]:
        st.markdown("""
        <div class="fun-fact-box">
        <h3>🌳 Green Field</h3>
        <div class="fun-fact-stat">27% Share</div>
        <ul class="fun-fact-bullets">
            <li>Share of European venture capital going to climate/green tech.</li>
            <li>Signals rising investor focus on decarbonisation and resilience.</li>
            <li>Momentum Trend: Climate is no longer “niche”.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with fact_cols[1]:
        st.markdown("""
        <div class="fun-fact-box">
        <h3>💰 Capital Powerhouse</h3>
        <div class="fun-fact-stat">5× Growth <span>(2016→2021)</span></div>
        <ul class="fun-fact-bullets">
            <li>VC grew from ~$20B to $100B+ in Europe.</li>
            <li>Expansion fueled bigger rounds + more late-stage activity.</li>
            <li>Set the baseline for today’s funding comparisons.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with fact_cols[2]:
        st.markdown("""
        <div class="fun-fact-box">
        <h3>🏙️ Beyond Borders</h3>
        <div class="fun-fact-stat">Top hubs <span>(London, Paris, Berlin)</span></div>
        <ul class="fun-fact-bullets">
            <li>These three dominate in scale and consistency.</li>
            <li>Strong “Next-Tier”: Switzerland, Sweden, Netherlands.</li>
            <li>Innovation clusters are spreading across Europe.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.write("Source: https://www.eu-startups.com/2025/08/10-years-that-changed-european-vc-trends-sectors-and-the-road-ahead/")

# Page: Geographical Insights (country_app.py)
@st.cache_data
def load_startups_country_year():
    query = """
    SELECT
        c.id,
        c.start_year,
        t.country_code,
        r.country
    FROM company c
    INNER JOIN city t ON c.city_id = t.city_id
    INNER JOIN country r ON t.country_code = r.country_code;
    """
    df = run_sql(query)

    df_country_year = df[(df["start_year"] >= 2014) & (df["start_year"] <= 2024)].copy()
    df_country_year = df_country_year.rename(columns={"country_code": "iso_alpha"})
    df_country_year["iso_alpha"] = df_country_year["iso_alpha"].astype(str)
    df_country_year = df_country_year.dropna(subset=["iso_alpha"])
    return df_country_year

@st.cache_data
def load_investment_data():
    query = """
    SELECT
        ci.id,
        ci.business_stage,
        ci.country_code,
        ci.year,
        ci.amount,
        c.country
    FROM country_investments ci
    INNER JOIN country c ON ci.country_code = c.country_code
    WHERE ci.business_stage != 'Total';
    """
    df = run_sql(query)

    if not df.empty:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df.dropna(subset=["year", "amount"])
    return df

def render_geo():
    safe_title("Geographical Insights")

    # --- Startups Founded map ---
    st.header("Startups across Europe")
    st.markdown("Use the slider to select a single year.")

    df_data = load_startups_country_year()

    MIN_YEAR, MAX_YEAR = 2014, 2024
    YEARS = list(range(MIN_YEAR, MAX_YEAR + 1))
    selected_year = st.select_slider("Select Year:", options=YEARS, value=MIN_YEAR)

    st.subheader(f"Startups Founded in **{selected_year}**")

    if df_data.empty:
        st.warning("No startup founding data available.")
    else:
        df_filtered = df_data[df_data["start_year"] == selected_year]
        total_startups_in_year = len(df_filtered)

        df_map = (
            df_filtered.groupby(["country", "iso_alpha"], as_index=False)["id"]
            .count()
            .rename(columns={"id": "Total Startups in Year"})
        )

        if df_map.empty:
            st.warning("No startups were founded in the selected year (according to this dataset).")
        else:
            df_map["Percentage"] = (df_map["Total Startups in Year"] / max(total_startups_in_year, 1)) * 100
            df_map["Percentage_Display"] = df_map["Percentage"].round(2).astype(str) + "%"

            fig = px.choropleth(
                df_map,
                locations="iso_alpha",
                color="Total Startups in Year",
                hover_name="country",
                hover_data={"Total Startups in Year": True, "Percentage_Display": True, "iso_alpha": False},
                color_continuous_scale=px.colors.sequential.Teal,
                scope="europe",
            )

            fig.update_traces(
                hovertemplate="<b>%{hovertext}</b><br>"
                              "Startups Founded: %{z:,}<br>"
                              "Percentage of startups founded: %{customdata[1]}<br>"
                              "<extra></extra>",
                customdata=df_map[["Total Startups in Year", "Percentage_Display"]].values,
                selector=dict(type="choropleth"),
            )
            fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0}, height=600)
            fig.update_geos(
                showocean=True,
                oceancolor="LightBlue",
                showframe=False,
                showcoastlines=True,
                projection_type="mercator",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown(f"**Total Startups in Selected Year:** **{total_startups_in_year:,}**")
            st.dataframe(
                df_map.sort_values("Total Startups in Year", ascending=False)
                     .set_index("country")[["Total Startups in Year", "Percentage_Display"]],
                use_container_width=True,
            )

    # --- Investment trends by stage ---
    st.markdown("---")
    st.header("Investment Trends by Business Stage")

    df_investment = load_investment_data()
    if df_investment.empty:
        st.warning("Investment data loading failed or the DataFrame is empty.")
        return

    CHART_MIN_YEAR, CHART_MAX_YEAR = 2007, 2024
    FULL_YEAR_RANGE = list(range(CHART_MIN_YEAR, CHART_MAX_YEAR + 1))

    all_countries = sorted(df_investment["country"].unique().tolist())
    col1, col2 = st.columns(2)

    with col1:
        selected_country = st.selectbox(
            "Select a Country:",
            options=all_countries,
            index=all_countries.index("Austria") if "Austria" in all_countries else 0
        )
    with col2:
        display_mode = st.selectbox("Display Mode:", options=["absolute", "percentage"])

    df_country = df_investment[df_investment["country"] == selected_country].copy()
    if df_country.empty:
        st.warning(f"No investment data available for **{selected_country}**.")
        return

    df_agg = df_country.groupby(["year", "business_stage"])["amount"].sum().reset_index()

    # Fill missing years/stages
    all_stages = df_investment["business_stage"].unique().tolist()
    df_full_years = pd.DataFrame({"year": FULL_YEAR_RANGE})
    df_full_stages = pd.DataFrame({"business_stage": all_stages})
    df_cross = df_full_years.assign(key=1).merge(df_full_stages.assign(key=1), on="key").drop("key", axis=1)

    df_chart = df_cross.merge(df_agg, on=["year", "business_stage"], how="left").fillna(0)

    df_total = df_chart.groupby("year")["amount"].sum().reset_index().rename(columns={"amount": "total_amount_year"})
    df_chart = df_chart.merge(df_total, on="year", how="left")
    df_chart["percentage"] = np.where(df_chart["total_amount_year"] > 0,
                                      (df_chart["amount"] / df_chart["total_amount_year"]) * 100,
                                      0)

    if display_mode == "absolute":
        y_column = "amount"
        title_text = f"Absolute Investment Trends in {selected_country} by Business Stage"
        y_label = "Investment Amount (Millions)"
        hover_fmt = "million $%{y:,.0f}"
    else:
        y_column = "percentage"
        title_text = f"Percentage Investment Trends in **{selected_country}** by Business Stage"
        y_label = "Investment Percentage"
        hover_fmt = "%{y:.2f}%"

    fig_bar = px.bar(
        df_chart,
        x="year",
        y=y_column,
        color="business_stage",
        title=title_text,
        labels={"year": "Year", y_column: y_label},
        color_discrete_sequence=px.colors.qualitative.Bold,
        category_orders={"year": FULL_YEAR_RANGE},
    )

    if display_mode == "percentage":
        fig_bar.update_layout(yaxis=dict(range=[0, 100], ticksuffix="%"))

    fig_bar.update_traces(
        hovertemplate=f"<b>Stage:</b> %{{fullData.name}}<br><b>Year:</b> %{{x}}<br><b>Investment:</b> {hover_fmt}<extra></extra>"
    )

    fig_bar.update_xaxes(
        tickmode="array",
        tickvals=FULL_YEAR_RANGE,
        ticktext=[str(y) for y in FULL_YEAR_RANGE],
        title_text="Year",
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        """
        1. **Seed:** Early capital used to validate the idea and build the first version.  
        2. **Start-up and other early stage:** Capital to scale (often Series A/B).  
        3. **Later stage venture:** Large rounds for expansion and exit preparation (Series C+).  
        """
    )

# Page: Funding Insights + Survivorship (fundings_longivity.py)
def fit_trendline(df: pd.DataFrame):
    X = df["start_year"].values.reshape(-1, 1)
    y = df["avg_funding"].values
    model = LinearRegression()
    model.fit(X, y)
    return model.predict(X), model.coef_[0], model.intercept_

def render_funding():
    safe_title("Funding Insights (2015–2024)")

    query1 = """
    SELECT start_year,
           COUNT(*) AS num_companies,
           AVG(funding_amount) AS avg_funding
    FROM company
    WHERE funding_amount IS NOT NULL AND start_year IS NOT NULL AND funding_amount <> 'NaN'
      AND start_year >= 2020 AND start_year < 2025
    GROUP BY start_year
    ORDER BY start_year ASC;
    """
    df1 = run_sql(query1)

    query2 = """
    SELECT start_year,
           COUNT(*) AS num_companies,
           AVG(funding_amount) AS avg_funding
    FROM company
    WHERE funding_amount IS NOT NULL AND start_year IS NOT NULL AND funding_amount <> 'NaN'
      AND start_year >= 2015 AND start_year < 2021
    GROUP BY start_year
    ORDER BY start_year ASC;
    """
    df2 = run_sql(query2)

    st.header("Fundings for startups in the last decade.")
    st.write("""
    This compares average funding across two periods (2015–2020 vs 2020–2024) with trendlines
    to highlight structural shifts in the funding environment.
    """)

    if df1.empty or df2.empty:
        st.warning("Not enough funding data for the trend comparison.")
        return

    trend1, slope1, _ = fit_trendline(df1)
    trend2, slope2, _ = fit_trendline(df2)

    df1["avg_funding_m"] = df1["avg_funding"] / 1e6
    df2["avg_funding_m"] = df2["avg_funding"] / 1e6
    trend1_m = trend1 / 1e6
    trend2_m = trend2 / 1e6

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df1["start_year"], df1["avg_funding_m"], marker="o", label="Average funding (2020+)", alpha=0.8)
    ax.plot(df2["start_year"], df2["avg_funding_m"], marker="x", label="Average funding (2015–2020)", alpha=0.8)
    ax.plot(df1["start_year"], trend1_m, linestyle="--", linewidth=2, label="Trendline (2020+)")
    ax.plot(df2["start_year"], trend2_m, linestyle="--", linewidth=2, label="Trendline (2015–2020)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Funding amount in million (€)")
    ax.set_title("Trend of fundings for startups in the last decade")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    st.pyplot(fig)

    st.subheader("Overall funding trends over the last decade.")
    st.write(f"2015–2020 trend slope: {round(slope2/1e6, 2):,.2f} million per year.")
    st.write(f"2020–2024 trend slope: {round(slope1/1e6, 2):,.2f} million per year.")

    # Growth clusters by city
    st.markdown("---")
    st.header("Startup growth clusters by City (2015–2024)")

    query_growth_cities = """
    SELECT 
        CASE 
            WHEN ci.city_name IN ('Vienna') THEN 'Vienna'
            WHEN ci.city_name IN ('London') THEN 'London'
            ELSE ci.city_name
        END AS city_name,
        c.start_year, 
        COUNT(*) AS num_companies, 
        AVG(c.funding_amount) AS avg_funding
    FROM company c
    JOIN city ci ON c.city_id = ci.city_id
    WHERE ci.city_name IN ('Vienna', 'London', 'Berlin', 'Paris', 'Barcelona')
      AND c.funding_amount IS NOT NULL and c.funding_amount <> 'NaN'
      AND c.start_year IS NOT NULL
      AND c.start_year > 2014 AND c.start_year < 2025
    GROUP BY city_name, c.start_year
    ORDER BY city_name, c.start_year;
    """
    df_growth = run_sql(query_growth_cities)

    if df_growth.empty:
        st.warning("No city cluster data available.")
        return

    df_growth["avg_funding_m"] = df_growth["avg_funding"] / 1e6

    fig_companies = px.line(
        df_growth,
        x="start_year",
        y="num_companies",
        color="city_name",
        markers=True,
        title="Number of companies over the years",
        labels={"num_companies": "Number of Companies", "start_year": "Year", "city_name": "City"},
    )
    fig_companies.update_layout(template="plotly_white", xaxis=dict(dtick=1))
    fig_companies.add_vline(x=2020, line_width=2, line_dash="dash", line_color="red",
                            annotation_text="COVID start (2020)", annotation_position="top left")
    fig_companies.add_vline(x=2023, line_width=2, line_dash="dash", line_color="red",
                            annotation_text="COVID end (2023)", annotation_position="top right")
    st.plotly_chart(fig_companies, use_container_width=True)

    fig_funding = px.line(
        df_growth,
        x="start_year",
        y="avg_funding_m",
        color="city_name",
        markers=True,
        title="Average funding amount for startup clusters by city over time",
        labels={"avg_funding_m": "Funding amount in million (€)", "start_year": "Year", "city_name": "City"},
    )
    fig_funding.update_layout(template="plotly_white", xaxis=dict(dtick=1))
    fig_funding.add_vline(x=2020, line_width=2, line_dash="dash", line_color="red",
                          annotation_text="COVID start (2020)", annotation_position="top left")
    fig_funding.add_vline(x=2023, line_width=2, line_dash="dash", line_color="red",
                          annotation_text="COVID end (2023)", annotation_position="top right")
    st.plotly_chart(fig_funding, use_container_width=True)

    # Industry analysis per city
    st.markdown("---")
    st.subheader("Top industries by funding per city")

    query_industry = """
    SELECT city_name, industry_tag, avg_funding
    FROM (
        SELECT
            ci.city_name,
            it.industry_tag,
            SUM(c.funding_amount) AS avg_funding,
            ROW_NUMBER() OVER (
                PARTITION BY ci.city_name
                ORDER BY AVG(c.funding_amount) DESC
            ) AS rn
        FROM company c
        JOIN company_industries ci2 ON c.id = ci2.company_id
        JOIN industry_tags it ON ci2.industry_id = it.industry_id
        JOIN city ci ON c.city_id = ci.city_id
        WHERE c.funding_amount IS NOT NULL AND c.funding_amount <> 'NaN'
          AND ci.city_name IN ('Vienna', 'Berlin', 'London', 'Paris', 'Barcelona')
          AND c.start_year IS NOT NULL
          AND c.start_year > 2014 AND c.start_year < 2025
        GROUP BY ci.city_name, it.industry_tag
    ) sub
    WHERE rn <= 1
    ORDER BY city_name, avg_funding DESC;
    """
    df_ind = run_sql(query_industry)
    if not df_ind.empty:
        fig = px.bar(
            df_ind,
            y="industry_tag",
            x="avg_funding",
            color="city_name",
            barmode="group",
            orientation="h",
            labels={"avg_funding": "Funding amount in €", "industry_tag": "Industry", "city_name": "City"},
            title="Top industries by funding per city",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# Survivorship
def render_survivorship():
    safe_title("Survivorship (2015–2024)")

    query_survivorship = """
    SELECT
        start_year,
        COUNT(*) AS total_startups,
        SUM(CASE WHEN (start_year + 3 <= EXTRACT(YEAR FROM CURRENT_DATE)) AND s.is_active THEN 1 ELSE 0 END) AS active_after_3_years,
        SUM(CASE WHEN (start_year + 5 <= EXTRACT(YEAR FROM CURRENT_DATE)) AND s.is_active THEN 1 ELSE 0 END) AS active_after_5_years,
        SUM(CASE WHEN (start_year + 6 <= EXTRACT(YEAR FROM CURRENT_DATE)) AND s.is_active THEN 1 ELSE 0 END) AS active_after_6_years,
        SUM(CASE WHEN (start_year + 7 <= EXTRACT(YEAR FROM CURRENT_DATE)) AND s.is_active THEN 1 ELSE 0 END) AS active_after_7_years,
        SUM(CASE WHEN (start_year + 10 <= EXTRACT(YEAR FROM CURRENT_DATE)) AND s.is_active THEN 1 ELSE 0 END) AS active_after_10_years
    FROM company c
    JOIN status s ON c.status_id = s.status_id
    WHERE start_year IS NOT NULL AND start_year > 2014 AND start_year < 2025
    GROUP BY start_year
    ORDER BY start_year;
    """
    df_surv = run_sql(query_survivorship)

    if df_surv.empty:
        st.warning("No survivorship data available.")
        return

    total_startups = df_surv["total_startups"].sum()
    active_3 = df_surv["active_after_3_years"].sum()
    active_5 = df_surv["active_after_5_years"].sum()
    active_6 = df_surv["active_after_6_years"].sum()
    active_7 = df_surv["active_after_7_years"].sum()
    active_10 = df_surv["active_after_10_years"].sum()

    funnel = pd.DataFrame({
        "Stage": ["Founded", "Active after 3 years", "Active after 5 years", "Active after 6 years",
                  "Active after 7 years", "Active after 10 years"],
        "Count": [total_startups, active_3, active_5, active_6, active_7, active_10],
    })
    funnel["Percentage"] = (funnel["Count"] / max(total_startups, 1) * 100).round(1)
    funnel["Label"] = funnel["Percentage"].astype(str) + "%"

    st.header("Startup longevity overview")
    st.write("""
    This funnel shows how many startups remain active after different time horizons.
    It provides a simple read on survivorship using the status table.
    """)

    fig, ax = plt.subplots(figsize=(12, 6))
    highlight = {"Active after 6 years", "Active after 10 years"}
    colors = ["tab:blue" if s in highlight else "lightblue" for s in funnel["Stage"]]
    bars = ax.barh(funnel["Stage"], funnel["Count"], color=colors)
    ax.set_xlabel("Number of startups")
    ax.set_title("Company longevity funnel (2015–2024)")
    ax.invert_yaxis()

    for bar, label in zip(bars, funnel["Label"]):
        ax.text(
            bar.get_width() + max(funnel["Count"]) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center"
        )

    st.pyplot(fig)

# Page: Industry & NLP (industry_insights.py)
@st.cache_data
def load_industry_base():
    query = """
    SELECT
        c.id AS company_id,
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
    LEFT JOIN city ci ON c.city_id = ci.city_id
    LEFT JOIN country co ON ci.country_code = co.country_code
    LEFT JOIN status s ON c.status_id = s.status_id
    LEFT JOIN company_industries ci_map ON c.id = ci_map.company_id
    LEFT JOIN industry_tags it ON ci_map.industry_id = it.industry_id
    GROUP BY
        c.id, c.company_name, c.funding_amount, c.start_year, c.website,
        ci.city_name, co.country_code, co.country, s.is_active, s.status
    ORDER BY c.id;
    """
    df = run_sql(query)
    df["industries"] = df["industries"].apply(
        lambda x: [i for i in x if i is not None] if isinstance(x, (list, tuple)) else []
    )

    CURRENT_YEAR = datetime.now().year
    df["company_age"] = df["start_year"].apply(lambda y: CURRENT_YEAR - int(y) if pd.notna(y) else np.nan)
    df["has_funding"] = df["funding_amount"].notna()
    df["log_funding"] = df["funding_amount"].apply(lambda x: np.log1p(x) if pd.notna(x) and x > 0 else np.nan)
    df["n_industries"] = df["industries"].apply(lambda lst: len(lst) if isinstance(lst, list) else 0)
    return df

@st.cache_data
def build_industry_summary(df_industry_base: pd.DataFrame):
    df_industry_long = df_industry_base.explode("industries").copy()
    df_industry_long = df_industry_long[df_industry_long["industries"].notna() & (df_industry_long["industries"] != "")]

    industry_counts = (
        df_industry_long.groupby("industries")["company_id"].nunique()
        .reset_index(name="n_companies")
    )

    industry_funding_presence = (
        df_industry_long.assign(has_funding=df_industry_long["funding_amount"].notna())
        .groupby(["industries", "has_funding"])["company_id"].nunique()
        .unstack(fill_value=0)
        .rename(columns={False: "n_no_funding", True: "n_with_funding"})
        .reset_index()
    )

    df_funded = df_industry_base.dropna(subset=["funding_amount"])
    df_funded_long = df_funded.explode("industries").copy()
    df_funded_long = df_funded_long[df_funded_long["industries"].notna() & (df_funded_long["industries"] != "")]

    industry_funding_stats = (
        df_funded_long.groupby("industries")["funding_amount"]
        .agg(n_funded_companies="count", median_funding="median", mean_funding="mean", max_funding="max")
        .reset_index()
    )

    industry_summary = (
        industry_counts
        .merge(industry_funding_presence, on="industries", how="left")
        .merge(industry_funding_stats, on="industries", how="left")
    )

    ind = industry_summary.copy()
    for col in ["n_funded_companies", "median_funding", "mean_funding"]:
        if col in ind.columns:
            ind[col] = pd.to_numeric(ind[col], errors="coerce")

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
    df = pd.read_csv(csv_path)

    def parse_list(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else []
        except Exception:
            return []

    if "industries_list" in df.columns:
        df["industries_list"] = df["industries_list"].apply(parse_list)

    if "started_in" in df.columns:
        df["started_in"] = pd.to_numeric(df["started_in"], errors="coerce")

    if "funding_usd" in df.columns:
        df["funding_usd"] = pd.to_numeric(df["funding_usd"], errors="coerce")

    return df

@st.cache_data
def build_cluster_summaries(df_nlp: pd.DataFrame):
    df_clusters = df_nlp[df_nlp["cluster"] != -1].copy()
    clusters = sorted(df_clusters["cluster"].dropna().unique())

    topics = []
    for c in clusters:
        texts = df_clusters[df_clusters["cluster"] == c]["clean_text"].astype(str).tolist()
        if len(texts) < 3:
            continue
        vec = TfidfVectorizer(max_features=20)
        _ = vec.fit_transform(texts)
        keywords = vec.get_feature_names_out()
        topics.append({"cluster": c, "size": len(texts), "keywords": ", ".join(keywords[:7])})
    topic_df = pd.DataFrame(topics).sort_values("size", ascending=False)

    df_ci = df_clusters.explode("industries_list").copy()
    df_ci = df_ci[df_ci["industries_list"].notna()]
    df_ci = df_ci[df_ci["industries_list"].astype(str).str.strip() != ""]

    cluster_industry_counts = (
        df_ci.groupby(["cluster", "industries_list"]).size().reset_index(name="count")
    )

    cluster_funding = (
        df_ci[df_ci["funding_usd"].notna()]
        .groupby("cluster")["funding_usd"]
        .agg(n_funded="count", median_funding="median", mean_funding="mean", max_funding="max")
        .reset_index()
    )

    top_industry_per_cluster = (
        cluster_industry_counts.sort_values(["cluster", "count"], ascending=[True, False])
        .groupby("cluster").head(1)
        .rename(columns={"industries_list": "top_industry", "count": "top_industry_count"})
    )

    cluster_summary = (
        topic_df.merge(cluster_funding, on="cluster", how="left")
        .merge(top_industry_per_cluster, on="cluster", how="left")
        .sort_values("size", ascending=False)
    )

    return topic_df, df_ci, cluster_industry_counts, cluster_funding, cluster_summary

def render_industry():
    safe_title("Industry & Theme Analysis")

    st.sidebar.header("Industry/NLP Controls")
    nlp_csv_path = st.sidebar.text_input(
        "Path to `failory_nlp_clusters.csv`",
        value="/workspaces/dsi-ws2025-project-grpab-weigl-mds1ab-awp-proj2/5. Data Analysis/data/failory_nlp_clusters.csv"
    )

    df_industry_base = load_industry_base()

    countries = sorted(df_industry_base["country"].dropna().unique())
    selected_countries = st.sidebar.multiselect(
        "Filter countries (industry plots)",
        options=countries,
        default=countries
    )

    if selected_countries:
        df_industry_filtered = df_industry_base[df_industry_base["country"].isin(selected_countries)].copy()
    else:
        df_industry_filtered = df_industry_base.copy()

    industry_summary, ind_filtered = build_industry_summary(df_industry_filtered)

    df_nlp = None
    cluster_summary = None
    df_ci = None
    cluster_industry_counts = None

    if os.path.exists(nlp_csv_path):
        df_nlp = load_nlp_clusters(nlp_csv_path)
        topic_df, df_ci, cluster_industry_counts, cluster_funding, cluster_summary = build_cluster_summaries(df_nlp)
    else:
        st.sidebar.warning("NLP CSV not found. NLP tab will be limited.")

    tab1, tab2 = st.tabs(["Industry Overview", "NLP Theme Clusters"])

    with tab1:
        st.subheader("Industry Overview")
        st.markdown("""
        We focus on:
        - **Where startups concentrate by industry** (counts)
        - **Which industries attract higher typical funding** (median funding)
        - **How reliable funding coverage is** (funding_ratio)
        """)

        top_n = st.slider("Top N industries", 5, 40, 20, step=5)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top Industries by Company Count")
            top_size = industry_summary.sort_values("n_companies", ascending=False).head(top_n)
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.barplot(data=top_size, y="industries", x="n_companies", ax=ax)
            ax.set_xlabel("Number of Companies")
            ax.set_ylabel("Industry")
            st.pyplot(fig)

        with col2:
            st.markdown("#### Top Industries by Median Funding (Filtered)")
            top_median = ind_filtered.sort_values("median_funding", ascending=False).head(top_n)
            top_median = top_median.copy()
            top_median["median_funding_m"] = top_median["median_funding"] / 1e6
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.barplot(data=top_median, y="industries", x="median_funding_m", ax=ax)
            ax.set_xlabel("Median Funding (€M)")
            ax.set_ylabel("Industry")
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("#### Industry Size vs Median Funding (Bubble View)")

        bubble_df = ind_filtered.copy()
        bubble_df["median_funding_m"] = bubble_df["median_funding"] / 1e6  # Converting into millions
        fig, ax = plt.subplots(figsize=(9, 4.8))
        scatter = ax.scatter(
            bubble_df["n_companies"],
            bubble_df["median_funding_m"],
            s=np.clip(bubble_df["n_funded_companies"], 5, 300) * 1.6,
            c=bubble_df["funding_ratio"],
            cmap="viridis",
            alpha=0.7
        )
        ax.set_xlabel("Number of Companies")
        ax.set_ylabel("Median Funding (€M)")
        ax.set_title("Industry Size vs Median Funding")
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Share of Companies with Funding Data")
        plt.tight_layout()
        st.pyplot(fig, width='stretch')

    with tab2:
        st.subheader("NLP-Derived Startup Themes")

        if df_nlp is None or cluster_summary is None:
            st.info("NLP data not loaded. Provide the correct CSV path in the sidebar.")
            return

        cluster_ids = cluster_summary["cluster"].tolist()
        # Build readable labels for clusters
        cluster_labels = {
            row["cluster"]: f"Cluster {int(row['cluster'])}: {row['top_industry']}"
            for _, row in cluster_summary.iterrows()
        }

        selected_cluster = st.selectbox(
            "Select a cluster to inspect:",
            options=list(cluster_labels.keys()),
            format_func=lambda c: cluster_labels[c]
        )

        row = cluster_summary[cluster_summary["cluster"] == selected_cluster].iloc[0]
        st.markdown("### Cluster summary")
        st.write({
            "Cluster": int(row["cluster"]),
            "Size (# startups)": int(row["size"]),
            "Keywords": row["keywords"],
            "Top industry": row.get("top_industry"),
            "Top industry count": None if pd.isna(row.get("top_industry_count")) else int(row["top_industry_count"]),
            "Median funding": None if pd.isna(row.get("median_funding")) else float(row["median_funding"]),
            "Mean funding": None if pd.isna(row.get("mean_funding")) else float(row["mean_funding"]),
            "Max funding": None if pd.isna(row.get("max_funding")) else float(row["max_funding"]),
        })

        st.markdown("---")
        col1, col2 = st.columns([2, 1])

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
                "Clusters to show:",
                options=top_clusters,
                default=top_clusters[:5]
            )
            plot_df = cluster_year[cluster_year["cluster"].isin(selected_for_growth)].copy()

            fig, ax = plt.subplots(figsize=(10, 5))
            if not plot_df.empty:
                sns.lineplot(data=plot_df, x="started_in", y="n", hue="cluster", marker="o", ax=ax)
                ax.set_xlabel("Start Year")
                ax.set_ylabel("Number of Startups")
                ax.set_title("Cluster Growth Over Time")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.info("No data for selected clusters / years.")

        with col2:
            st.markdown("#### Wordcloud for Selected Cluster")
            subset = df_nlp[df_nlp["cluster"] == selected_cluster].copy()
            text_series = subset["clean_text"].dropna() if "clean_text" in subset.columns else subset["description"].astype(str).dropna()

            if not text_series.empty:
                big_text = " ".join(text_series.tolist())
                wc = WordCloud(width=600, height=400, background_color="white", max_words=100).generate(big_text)
                fig_wc, ax_wc = plt.subplots(figsize=(5, 3))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("No text available for wordcloud.")

        st.markdown("---")
        st.markdown("#### Top Clusters by Median Funding")
        top_k = st.slider("Show top K clusters", 3, 15, 10)

        top_funding = (
            cluster_summary.dropna(subset=["median_funding"])
            .assign(median_funding_m=lambda d: d["median_funding"] / 1e6)
            .sort_values("median_funding_m", ascending=False)
            .head(top_k)
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(data=top_funding, x="cluster", y="median_funding_m", palette=sns.color_palette("viridis", n_colors=len(top_funding)), ax=ax)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Median Funding (€M)")
        ax.set_title("Top Clusters by Median Funding")
        ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig)

# Page: Status & Lifecycle (status_analysis.py)
@st.cache_data
def load_status_data():
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
    df = run_sql(query)

    CURRENT_YEAR = datetime.now().year
    df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")
    df["start_year"] = df["start_year"].apply(lambda y: y if pd.notna(y) and 1990 <= y <= 2025 else np.nan)
    df["age"] = CURRENT_YEAR - df["start_year"]

    def map_lifecycle(row):
        if row["status"] == "active" or row["is_active"] is True:
            return "active"
        elif row["status"] == "inactive" or row["is_active"] is False:
            return "inactive"
        else:
            return "unknown"
    df["lifecycle_status"] = df.apply(map_lifecycle, axis=1)

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

    country_status = (
        df.groupby(["country", "lifecycle_status"])["company_id"]
        .count()
        .reset_index(name="n_companies")
    )
    total_per_country = (
        country_status.groupby("country")["n_companies"].sum()
        .reset_index(name="total_companies")
    )
    country_status = country_status.merge(total_per_country, on="country")
    country_status["ratio"] = country_status["n_companies"] / country_status["total_companies"]

    country_pivot = country_status.pivot_table(
        index="country", columns="lifecycle_status", values="ratio", fill_value=0
    ).reset_index()

    counts_per_country = (
        df.groupby("country")["company_id"].count().reset_index(name="n_companies")
    )
    country_pivot = country_pivot.merge(counts_per_country, on="country")

    for col in ["active", "inactive", "unknown"]:
        if col not in country_pivot.columns:
            country_pivot[col] = 0.0

    return df, lifecycle_summary, country_pivot

def render_status():
    safe_title("Status & Lifecycle")

    st.markdown(
        "This section explores **company age, status, funding, and country-level survival patterns** "
        "based on the unified dataset."
    )

    df, lifecycle_summary, country_pivot = load_status_data()
    if df.empty:
        st.warning("No data available.")
        return

    st.sidebar.header("Filters")
    countries = sorted(df["country"].dropna().unique())
    selected_countries = st.sidebar.multiselect("Countries", options=countries, default=countries)

    status_options = ["active", "inactive", "unknown"]
    selected_status = st.sidebar.multiselect("Lifecycle status", options=status_options, default=status_options)

    min_age = int(np.nanmin(df["age"]))
    max_age = int(np.nanmax(df["age"]))
    age_range = st.sidebar.slider("Age range (years)", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    mask = (
        df["country"].isin(selected_countries)
        & df["lifecycle_status"].isin(selected_status)
        & df["age"].between(age_range[0], age_range[1], inclusive="both")
    )
    df_filtered = df[mask].copy()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Startups in selection", f"{len(df_filtered):,}")
    with col2:
        active_share = (df_filtered["lifecycle_status"] == "active").mean() if len(df_filtered) else 0
        st.metric("Active share", f"{active_share:.1%}")
    with col3:
        st.metric("Median age (years)", f"{df_filtered['age'].median():.1f}")
    with col4:
        st.metric("Funding coverage", f"{df_filtered['funding_amount'].notna().mean():.1%}")

    st.markdown("---")

    st.subheader("Lifecycle Status Breakdown")
    status_counts = (
        df_filtered["lifecycle_status"].value_counts()
        .reindex(status_options).fillna(0).reset_index()
    )
    status_counts.columns = ["lifecycle_status", "n"]

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.barplot(data=status_counts, x="lifecycle_status", y="n", ax=ax)
    ax.set_xlabel("Lifecycle Status")
    ax.set_ylabel("Number of Companies")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Age Distribution of Companies")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=df_filtered, x="age", bins=40, kde=True, ax=ax)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of Companies")
    st.pyplot(fig)

    st.subheader("Company Age by Lifecycle Status")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_filtered, x="lifecycle_status", y="age", order=status_options, ax=ax)
    ax.set_xlabel("Lifecycle Status")
    ax.set_ylabel("Age (years)")
    st.pyplot(fig)

    st.subheader("Funding & Survival")
    st.dataframe(
        lifecycle_summary.style.format({
            "n_companies": "{:,.0f}",
            "median_age": "{:.1f}",
            "mean_age": "{:.1f}",
            "n_with_funding": "{:,.0f}",
            "median_funding": "{:,.0f}",
            "mean_funding": "{:,.0f}",
        }),
        use_container_width=True,
    )

    df_funded = df_filtered[df_filtered["funding_amount"].notna()]
    df_funded["funding_m"] = df_funded["funding_amount"] / 1e11
    if not df_funded.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df_funded, x="age", y="funding_m", hue="lifecycle_status", hue_order=status_options, alpha=0.6, ax=ax)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Funding (€M)")
        st.pyplot(fig)

# Main router
PAGES = {
    "Home": render_home,
    "Geographical Insights": render_geo,
    "Funding Insights": render_funding,
    "Survivorship": render_survivorship,
    "Industry & NLP Themes": render_industry,
    "Status & Lifecycle": render_status,
}

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", list(PAGES.keys()), index=0)
PAGES[page]()