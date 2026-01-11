import streamlit as st
import psycopg2
from dotenv import load_dotenv
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

load_dotenv()

# --- 1. Database Connection and Data Loading ---

@st.cache_data
def get_connection_params():
    return dict(
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname"),
        user=os.getenv("user"),
        password=os.getenv("password"),
    )

### FOR STARTUP FOUNDING DATA BY COUNTRY ###############################################

@st.cache_data
def load_and_prepare_data():
    params = get_connection_params()
    try:
        conn = psycopg2.connect(**params)
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        # Return an empty DataFrame to prevent further execution errors
        return pd.DataFrame(columns=['id', 'start_year', 'country_code', 'country'])


    query = """
    SELECT
        c.id,
        c.start_year,
        t.country_code, 
        r.country
    FROM
        company c
    INNER JOIN
        city t ON c.city_id = t.city_id
    INNER JOIN
        country r ON t.country_code = r.country_code;
        """
    df = pd.read_sql(query, conn)
    conn.close()

    # Filter for years that have enough startup data
    df_country_year = df[
        (df['start_year'] >= 2014) & 
        (df['start_year'] <= 2024)
    ].copy()

    df_country_year = df_country_year.rename(columns={'country_code': 'iso_alpha'})
    df_country_year['iso_alpha'] = df_country_year['iso_alpha'].astype(str)
    df_country_year = df_country_year.dropna(subset=['iso_alpha'])
    
    return df_country_year

### FOR INVESTMENT BY STAGE PER COUNTRY ##############################################################

@st.cache_data
def load_investment_data():
    params = get_connection_params()
    try:
        conn = psycopg2.connect(**params)
    except Exception as e:
        st.error(f"Error connecting to investment database: {e}")
        return pd.DataFrame(columns=['business_stage', 'amount', 'country', 'year'])

    query = """
    SELECT
        ci.id,
        ci.business_stage,
        ci.country_code,
        ci.year,
        ci.amount,
        c.country
    FROM
        country_investments ci
    INNER JOIN
        country c ON ci.country_code = c.country_code
    WHERE 
        ci.business_stage != 'Total';
        """
    
    df_investment = pd.read_sql(query, conn)
    conn.close()

    if not df_investment.empty:
        df_investment['year'] = pd.to_numeric(df_investment['year'], errors='coerce').astype('Int64')
        df_investment['amount'] = pd.to_numeric(df_investment['amount'], errors='coerce')
        df_investment = df_investment.dropna(subset=['year', 'amount'])
    
    return df_investment

# Load the data
df_data = load_and_prepare_data()
df_investment = load_investment_data()

### FOR COUNTRY FOUNDING DISPLAY #########################################################

# Streamlit UI and Slider 
st.title("Startups across Europe")
st.markdown("Use the slider to select a single year.")

# slider 
MIN_YEAR = 2014
MAX_YEAR = 2024

YEARS = list(range(MIN_YEAR, MAX_YEAR + 1)) 


selected_year = st.select_slider(
    "Select Year:",
    options=YEARS,
    value=MIN_YEAR # Default to the latest year (2024)
)


st.subheader(f"Startups Founded in **{selected_year}**")

# Data Filtering and Aggregation 

if df_data.empty:
    st.warning("Data loading failed or the returned DataFrame is empty.")
else:
    # filter based on user input
    df_filtered = df_data[
        (df_data['start_year'] >= selected_year) & 
        (df_data['start_year'] <= selected_year)
    ]

    # total startups
    total_startups_in_timespan = len(df_filtered)

    # aggregate

    df_map = (
        df_filtered.groupby(['country', 'iso_alpha'], as_index=False)['id']
        .count()
        .rename(columns={'id': 'Total Startups in Year'})
    )
    # percentage
    df_map['Percentage'] = (df_map['Total Startups in Year'] / total_startups_in_timespan) * 100
    df_map['Percentage_Display'] = df_map['Percentage'].round(2).astype(str) + '%'

    # Mapping
    
    if df_map.empty:
        st.warning("No startups were founded in the selected year.")
    else:
        fig = px.choropleth(
            df_map,
            locations='iso_alpha',           
            color='Total Startups in Year', 
            hover_name='country',            
            hover_data={
                'Total Startups in Year': True,  
                'Percentage_Display': True,        
                'iso_alpha': False                 
            },
            color_continuous_scale=px.colors.sequential.Teal, 
            scope='europe',
#            title=f'Global Distribution of New Startups: {selected_year}',
        )

        # hover template
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>" + 
                          "Startups Founded: %{z:,}<br>" +
                          "Percentage of startups founded: %{customdata[1]}<br>" +
                          "<extra></extra>",
            customdata=df_map[['Total Startups in Year', 'Percentage_Display']].values,
            selector=dict(type='choropleth')
        )

        # map layout adjustments
        fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, height=600)
        fig.update_geos(
            showocean=True,
            oceancolor='LightBlue',
            showframe=False,
            showcoastlines=True,
            projection_type="mercator"
        )
        
        # display plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the underlying data details
        st.markdown("---")
        st.markdown(f"**Total Startups in Selected Year:** **{total_startups_in_timespan:,}**")
        
        # Optional: Show the top countries in a table
        st.dataframe(
            df_map.sort_values('Total Startups in Year', ascending=False)
            .set_index('country')[['Total Startups in Year', 'Percentage_Display']], 
            use_container_width=True
        )

st.markdown("---")
st.title("Investment Trends by Business Stage")

if df_investment.empty:
    st.warning("Investment data loading failed or the DataFrame is empty.")
else:

    CHART_MIN_YEAR = 2007
    CHART_MAX_YEAR = 2024
    FULL_YEAR_RANGE = list(range(CHART_MIN_YEAR, CHART_MAX_YEAR + 1))

    # Interactive UI Elements for Investment Chart

    # all countries for dropdown
    all_countries = sorted(df_investment['country'].unique().tolist())
    
    default_country = all_countries[0] if all_countries else 'No Data'

    # dropdown
    col1, col2 = st.columns(2)

    with col1:
        selected_country = st.selectbox(
            "Select a Country:",
            options=all_countries,
            index=all_countries.index('Austria') if 'Austria' in all_countries else 0
        )

    with col2:
        display_mode = st.selectbox(
            "Display Mode:",
            options=['absolute', 'percentage']
        )
    
    # Data Filtering and Preprocessing 
    
    # Filter selected country
    df_country = df_investment[df_investment['country'] == selected_country].copy()

    if df_country.empty:
        st.warning(f"No investment data available for **{selected_country}**.")
    else:
        # Aggregate the investment amount by year and business stage
        df_agg = df_country.groupby(['year', 'business_stage'])['amount'].sum().reset_index()

        # Calculate the total investment for each year
        df_total_yearly = df_agg.groupby('year')['amount'].sum().reset_index().rename(columns={'amount': 'total_amount_year'})
        
        # Merge total yearly amount back into the aggregated dataframe
        df_agg = pd.merge(df_agg, df_total_yearly, on='year', how='right')

        # Calculate the percentage of total investment for each stage within the year
        df_agg['percentage'] = (df_agg['amount'] / df_agg['total_amount_year']) * 100
        df_agg['percentage'] = df_agg['percentage'].fillna(0) # Fill NaN from years with zero investment

        # Fill in Missing Years and Business Stages
        # not all countries have all years
        
        # Create a DataFrame with ALL years and ALL business stages 
        all_stages = df_investment['business_stage'].unique().tolist()
        df_full_years = pd.DataFrame({'year': FULL_YEAR_RANGE})
        df_full_stages = pd.DataFrame({'business_stage': all_stages})
        
        # Create  all years and all stages
        df_cross_join = df_full_years.assign(key=1).merge(
            df_full_stages.assign(key=1), on='key'
        ).drop('key', axis=1)

        # Merge the aggregated data (df_agg) with the full year/stage matrix
        # insert 0 for 'amount' and 'percentage' for years where a stage had no investment
        df_chart_data = pd.merge(
            df_cross_join, 
            df_agg, 
            on=['year', 'business_stage'], 
            how='left'
        ).fillna(0)
        
        # Re-calculate the percentage for the final chart data
        df_total_yearly_final = df_chart_data.groupby('year')['amount'].sum().reset_index().rename(columns={'amount': 'total_amount_year'})
        
        df_chart_data = pd.merge(
            df_chart_data.drop(columns=['total_amount_year', 'percentage'], errors='ignore'), 
            df_total_yearly_final, 
            on='year', 
            how='left'
        )
        df_chart_data['percentage'] = (df_chart_data['amount'] / df_chart_data['total_amount_year']) * 100
        df_chart_data['percentage'] = df_chart_data['percentage'].fillna(0) # Handle divide by zero for years with NO total investment

        # Create Interactive Stacked Bar Chart 
        
        if display_mode == 'absolute':
            y_column = 'amount'
            title_text = f"Absolute Investment Trends in **{selected_country}** by Business Stage"
            hovertemplate_amount = 'million $%{y:,.0f}'
        else: # percentage mode
            y_column = 'percentage'
            title_text = f"Percentage Investment Trends in **{selected_country}** by Business Stage"
            hovertemplate_amount = '%{y:.2f}%'

        fig_bar = px.bar(
            df_chart_data,
            x='year', 
            y=y_column,
            color='business_stage',
            title=title_text,
            labels={'year': 'Year', y_column: 'Investment Amount (Millions)' if display_mode == 'absolute' else 'Investment Percentage'},
            color_discrete_sequence=px.colors.qualitative.Bold,
            category_orders={'year': FULL_YEAR_RANGE} # years are ordered 
        )

        # percentage mode bars same height - spine chart
        if display_mode == 'percentage':
            fig_bar.update_layout(yaxis=dict(range=[0, 100], ticksuffix="%"))
            fig_bar.update_traces(
                hovertemplate=f"<b>Stage:</b> %{{fullData.name}}<br><b>Year:</b> %{{x}}<br><b>Investment:</b> {hovertemplate_amount}<extra></extra>",
                customdata=df_chart_data[['business_stage', 'amount']].values
            )
        else:
            #  absolute mode
             fig_bar.update_traces(
                hovertemplate=f"<b>Stage:</b> %{{fullData.name}}<br><b>Year:</b> %{{x}}<br><b>Investment:</b> {hovertemplate_amount}<extra></extra>",
                customdata=df_chart_data[['business_stage', 'amount']].values
            )


        # all years from 2007 to 2024 are on the x-axis
        fig_bar.update_xaxes(
            tickmode='array', 
            tickvals=FULL_YEAR_RANGE, 
            ticktext=[str(y) for y in FULL_YEAR_RANGE],
            title_text="Year"
        )
        
        # Display the plot 
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown(
            """
        1. **Seed:** The initial capital used to "plant the seed" for the business. It's often the first formal investment round after the founders' own money or funds from friends/family.
        2. **Start-up and other early stage:** Funding rounds (often Series A, then Series B) used to transition from a validated concept to a scalable business.
        3. **Later stage venture:** Larger funding rounds (Series C, D, etc.) designed for massive expansion and preparing for a potential "exit," such as an Initial Public Offering (IPO) or acquisition.
        """
        )