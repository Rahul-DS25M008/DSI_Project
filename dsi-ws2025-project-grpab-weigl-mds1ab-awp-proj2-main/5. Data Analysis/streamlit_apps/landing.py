import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="home",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom Styling for Visual Polish (Optional) ---
st.markdown("""
<style>
/* 1. Main Container Padding */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* 2. Style for Fun Fact Boxes */
.fun-fact-box {
    background-color: #f0f8ff; /* Light blue/off-white background */
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #007BFF; /* A striking blue accent border */
    margin-bottom: 20px;
    box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
}
.fun-fact-box h3 {
    color: #007BFF;
}

/* 3. Image Sizing/Styling (if needed, but st.image handles most) */
</style>
""", unsafe_allow_html=True)


# --- 1. Hero Section: Title and Mission ---
col_title, col_image = st.columns([2, 1])

with col_title:
    st.markdown("# **The European Startup Landscape Monitor** 🇪🇺")
    st.markdown("### Monitoring the pulse of innovation across the continent.")

with col_image:
    # Use an iconic image representing Europe or data
    st.image(
        "static/landing_page_1.jpg", 
    )
    # Note: Replace this image URL with your app's screenshot or a relevant, high-quality image.

st.markdown("---")


st.markdown("## **Fascinating Facts from the Start-up Ecosystem** ")

# Placeholder facts (you can replace the text/numbers after the tool search)
fact_cols = st.columns(3)

# Fact 1: Unicorns
with fact_cols[0]:
    st.markdown("""
    <div class="fun-fact-box">
        <h3>🌳 Green Field</h3>
        <p>Climate tech is gaining traction. In 2023, 27% of all venture capital investemnts made to green and climate tech start ups.</p>
    </div>
    """, unsafe_allow_html=True)

# Fact 2: Investment Total
with fact_cols[1]:
    st.markdown("""
    <div class="fun-fact-box">
        <h3>💰 Capital Powerhouse</h3>
        <p>From 2016 to 2021, Venture Capital investments grew 5-folds in Europe from $20 billion to over $100 billion.</p>
    </div>
    """, unsafe_allow_html=True)

# Fact 3: Top Hub
with fact_cols[2]:
    st.markdown("""
    <div class="fun-fact-box">
        <h3>🏙️ Beyond Borders</h3>
        <p>London, Paris, and Berlin consistently rank as the top three European cities. Innovation however doesn't stop there, Switzerland, Sweeden and the Netherlands are also developing a strong start-up ecosystem.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---") 

# --- 4. Footer and Contact (Static Information) ---
st.container()
st.write("Source: https://www.eu-startups.com/2025/08/10-years-that-changed-european-vc-trends-sectors-and-the-road-ahead/")
