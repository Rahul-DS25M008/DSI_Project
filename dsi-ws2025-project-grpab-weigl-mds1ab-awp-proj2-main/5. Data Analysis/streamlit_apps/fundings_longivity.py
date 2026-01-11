import streamlit as st
import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("host"),
    port=os.getenv("port"),
    dbname=os.getenv("dbname"),
    user=os.getenv("user"),
    password=os.getenv("password"),
)

# ----------------------------
# Streamlit App Title
# ----------------------------
# 
# Menu point "Fundings"
# st.title("Funding Analysis (2015–2024)")
#
st.title("Funding & Survivorship Analysis (2015–2024)")

# ----------------------------
# Funding Trends
# ----------------------------
query1 = """
SELECT start_year,
       COUNT(*) AS num_companies,
       AVG(funding_amount) AS avg_funding
FROM company
WHERE funding_amount IS NOT NULL AND start_year IS NOT NULL AND funding_amount <> 'NaN' AND start_year >= 2020
and start_year < 2025
GROUP BY start_year
ORDER BY start_year ASC;
"""

df1 = pd.read_sql(query1, conn)

query2 = """
SELECT start_year,
       COUNT(*) AS num_companies,
       AVG(funding_amount) AS avg_funding
FROM company
WHERE funding_amount IS NOT NULL AND start_year IS NOT NULL AND funding_amount <> 'NaN' AND start_year >= 2015 AND start_year < 2021
GROUP BY start_year
ORDER BY start_year ASC;
"""

df2 = pd.read_sql(query2, conn)

# Trendline function
def fit_trendline(df):
    X = df['start_year'].values.reshape(-1,1)
    y = df['avg_funding'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return y_pred, model.coef_[0], model.intercept_

trend1, slope1, intercept1 = fit_trendline(df1)
trend2, slope2, intercept2 = fit_trendline(df2)

df1['avg_funding_m'] = df1['avg_funding'] / 1e6
df2['avg_funding_m'] = df2['avg_funding'] / 1e6
trend1_m = trend1 / 1e6
trend2_m = trend2 / 1e6

# ----------------------------
# Plot Trends
# ----------------------------
st.header("Fundings for startups in the last decade.")
st.write("""
This following diagramm shows how average startup funding has evolved over the past decade by comparing two distinct periods: 2015–2020 and 2020 onward. 

The orange line represents the years 2015 to 2020, showing a gradual trend in funding levels before the pandemic. 
The blue line focuses on the years from 2020 onward, capturing the funding environment during and after the COVID-19 period. 

For both time ranges, dotted trendlines illustrate the overall direction of funding amounts. 
Together, the trajectories highlight how funding patterns shifted between a relatively stable pre-2020 market and a more volatile and evolving post-2020 landscape, making it clear how investor behavior and economic conditions have influenced startup financing over time.
""")

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(df1['start_year'], df1['avg_funding_m'], marker='o', color='blue', label='Average funding (2020+)', alpha=0.6)
ax.plot(df2['start_year'], df2['avg_funding_m'], marker='x', color='orange', label='Average funding (2015-2020)', alpha=0.6)

ax.plot(df1['start_year'], trend1_m, color='blue', linestyle='--', label='Trendline (2020+)', linewidth=2)
ax.plot(df2['start_year'], trend2_m, color='orange', linestyle='--', label='Trendline (2015-2020)', linewidth=2)

ax.set_xlabel('Year')
ax.set_ylabel('Funding amount in million (€)')
ax.set_title('Trend of fundings for startups in the last decade.')
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 5))

all_years = sorted(set(df1['start_year']).union(set(df2['start_year'])))
ax.set_xticks(all_years)

ax.legend()
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

st.pyplot(fig)

# ----------------------------
# Insights
# ----------------------------
st.subheader("Overall funding trends over the last decade.")

st.write(f"For the first half of the decade funding amount decreases over time more rapidly. For the first period of time (2015-2020) trend slope shows = {round(slope2/1000000,2):,.2f} million.") 
st.write(f"For the second half of the decade funding amount also decreases over time. However compared to the first time period, the trend slope shows a slower decrease. Trend slope shows = {round(slope1/1000000,2):,.2f} million.\n\n\n\n\n")


# ----------------------------
# Growth clusters by city
# ----------------------------
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
  AND c.start_year > 2014 
  AND c.start_year < 2025
GROUP BY city_name, c.start_year
ORDER BY city_name, c.start_year;
"""

df_growth_cities = pd.read_sql(query_growth_cities, conn)
df_growth_cities['total_funding_m'] = df_growth_cities['avg_funding'] / 1e6

st.header("Startup growth clusters by City (2015–2024)")
st.write("""
         This subset of the data focuses on the growth of startups in selected major cities — Vienna, London, Berlin, Paris, and Barcelona — for companies founded after 2014. 
         For each city and year, it reports the number of startups founded and their average funding amounts. 

         The data allows us to analyze growth clusters by city, showing which cities are most active in terms of new company formation and how funding trends differ across locations. 
         """)

fig_companies = px.line(
    df_growth_cities,
    x='start_year',
    y='num_companies',
    color='city_name',
    markers=True,
    title='Number of companies over the years',
    labels={'num_companies':'Number of Companies', 'start_year':'Year', 'city_name':'City'}
)
fig_companies.update_layout(
    template='plotly_white',
    xaxis=dict(dtick=1),
    title={
        'text': "Number of companies over the years<br><sup>Growth clusters of top 5 cities with the most number of companies in the dataset.</sup>",
        'font': {'size': 20}  
    })
fig_companies.add_vline(
    x=2020,
    line_width=2,
    line_dash="dash",
    line_color="red",
    annotation_text="COVID start (2020)",
    annotation_position="top left"
)

fig_companies.add_vline(
    x=2023,
    line_width=2,
    line_dash="dash",
    line_color="red",
    annotation_text="COVID end (2023)",
    annotation_position="top right"
)
st.plotly_chart(fig_companies)

# Funding chart
fig_funding = px.line(
    df_growth_cities,
    x='start_year',
    y='total_funding_m',
    color='city_name',
    markers=True,
    title='Average funding amount for startup clusters by city over time.',
    labels={'total_funding_m':'Funding amount in million (€)', 'start_year':'Year', 'city_name':'City'}
)
fig_funding.update_layout(template='plotly_white', xaxis=dict(dtick=1),title={'font': {'size': 20}})
fig_funding.add_vline(
    x=2020,
    line_width=2,
    line_dash="dash",
    line_color="red",
    annotation_text="COVID start (2020)",
    annotation_position="top left"
)

fig_funding.add_vline(
    x=2023,
    line_width=2,
    line_dash="dash",
    line_color="red",
    annotation_text="COVID end (2023)",
    annotation_position="top right"
)
st.plotly_chart(fig_funding)

# ----------------------------
# Industry analysis
# ----------------------------
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
    WHERE c.funding_amount IS NOT NULL
      AND c.funding_amount <> 'NaN'
      AND ci.city_name IN ('Vienna', 'Berlin', 'London', 'Paris', 'Barcelona')
      AND c.start_year IS NOT NULL
      AND c.start_year > 2014
      AND c.start_year < 2025
    GROUP BY ci.city_name, it.industry_tag
) sub
WHERE rn <= 1
ORDER BY city_name, avg_funding DESC;
"""

df_industry = pd.read_sql(query_industry, conn)
df_top3_industries = df_industry.groupby("city_name").head(6)

#st.header("Top industries by average funding per city")
st.write("""\n\n\n
         Both graphs above show a negative trend. This indicates that, according to our dataset, over the last decade the number of startups and their funding amounts have decreased in major cities such as Vienna, Berlin, Barcelona, Paris, and London.
         \nFurthermore, we explore the data and highlight which industries are driving the most investment in each of these ceties for the time period.""")


fig = px.bar(
    df_top3_industries,
    y='industry_tag',
    x='avg_funding',
    color='city_name',
    barmode='group',
    orientation='h',
    labels={
        'avg_funding': 'Average funding amount in €',
        'industry_tag': 'Industry',
        'city_name': 'City'
    }
)

fig.update_layout(
    title={'text': 'Top industries by average funding per city.','font': {'size': 20} },
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)

st.write("""\n\n\n
         As a result we have the following top 5 industries:""")
st.markdown("""
- IT Infrastructure: Companies providing hardware, networking, servers, data centers, and core systems that support IT operations.<br>
- Cloud Data Services: Companies providing cloud-based storage, computing, and analytics solutions.<br>
- Email Hosting Services: Providers delivering secure, reliable email services, including hosting, filtering, and collaboration tools.<br>
- InsurTech: Startups innovating in insurance using technology, such as AI-driven underwriting and claims automation.<br>
- Food Industry: Businesses involved in food production, processing, distribution, or technology.
""", unsafe_allow_html=True)









# ----------------------------
# Funnel
# ----------------------------
# 
# Menu point "Survivorship"
# st.title("Survivorship Analysis (2015–2024)")
#

# Startup survivorship data
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

df_survivorship = pd.read_sql(query_survivorship, conn)

total_startups = df_survivorship['total_startups'].sum()
active_3 = df_survivorship['active_after_3_years'].sum()
active_5 = df_survivorship['active_after_5_years'].sum()
active_6 = df_survivorship['active_after_6_years'].sum()
active_7 = df_survivorship['active_after_7_years'].sum()
active_10 = df_survivorship['active_after_10_years'].sum()

funnel_data = pd.DataFrame({
    'Stage': [
        'Founded',
        'Active after 3 years',
        'Active after 5 years',
        'Active after 6 years',
        'Active after 7 years',
        'Active after 10 years'
    ],
    'Count': [total_startups, active_3, active_5, active_6, active_7, active_10]
})

funnel_data['Percentage'] = (funnel_data['Count'] / total_startups * 100).round(1)
funnel_data['Label'] = funnel_data['Percentage'].astype(str) + '%'

st.header("Startup longivity overview")

st.write("""
This chart visualises startup survivorship over time. For the time period since 2015, it shows:

- The total number of startups founded.  
- How many of those are still active after different time periods.  

It allows us to understand the longevity and survival patterns of startups in the dataset.
""")
# Plot
fig, ax = plt.subplots(figsize=(12, 6))
#bars = ax.barh(funnel_data['Stage'], funnel_data['Count'], color='skyblue')
highlight_stages = ['Active after 6 years', 'Active after 10 years']
colors = [
    'tab:blue' if stage in highlight_stages else 'lightblue'
    for stage in funnel_data['Stage']
]
bars = ax.barh(
    funnel_data['Stage'],
    funnel_data['Count'],
    color=colors
)


ax.set_xlabel('Number of startups')
ax.set_title('Company longivity funnel (2015–2024)')
ax.invert_yaxis()  # founded on top

# Add % labels
for bar, label in zip(bars, funnel_data['Label']):
    ax.text(
        bar.get_width() + max(funnel_data['Count']) * 0.02,
        bar.get_y() + bar.get_height()/2,
        label,
        va='center'
    )

st.pyplot(fig)

st.write("""
From the funnel, we can see that after 6 years the number of active startups is almost half of the number of founded ones.
By the 10th year, only 5.9% of the companies in our dataset managed to survive over this time period.
""")