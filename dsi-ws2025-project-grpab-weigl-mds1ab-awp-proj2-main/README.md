[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vlj_khYU)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=20893838)

# DSI Project: The EU Startup Landscape Monitor

This repository contains the complete workflow for the DSI project, from raw data acquisition to a final interactive Streamlit application. Our project focuses on exploring the startups and company scenario across EU to draw some insights on industries, funding and the general outlook.

---

## Workflow Overview (Recommended)
For a high-level view of the end-to-end pipeline, refer to:

- **[Workflow_EU_Startup_Monitor.xlsx](Workflow_EU_Startup_Monitor.xlsx)** – Project workflow map and execution checklist

---

## Environment Setup

- **`requirements.txt`**  
  Documents all libraries and frameworks used throughout the project for tasks.  
  Install this once to skip using virtual environment.

- **`.venv/`**  
  Pre-configured virtual environment with all required dependencies installed.  
  Use this environment for running notebooks and applications.

---

## Project Structure

### 1. Data Acquisition
The first stage focuses on collecting data from multiple sources using methods such as:
- Web scraping  
- API access  

**Structure:**
- `notebooks/` – Reproducible scripts used for data collection  
- `data/` – Raw datasets obtained from the sources  

---

### 2. Data Cleaning
Each team member cleans the data they collected individually.

**Structure:**
- `notebooks/` – Scripts for data cleaning and preprocessing  
- `data/` – Cleaned datasets  

---

### 3. Data Merging
This stage consists of two steps:
1. **Standardization** of individual datasets based on an agreed schema  
2. **Merging** all standardized datasets into a single clean dataset  

**Structure:**
- `notebooks/` – Standardization and merging scripts  
- `data/` – Final merged dataset  

---

### 4. Database Transfer
This step handles database design and data ingestion.

**Tasks include:**
- Designing the ER schema  
- Creating tables and relationships  
- Transferring cleaned data into the database  

**Structure:**
- `Schema Creation/` – ER diagrams and schema definitions  
- `notebooks/` – Scripts for database creation and data insertion  
- `data/` – Final datasets with correct data types and foreign keys  

---

### 5. Data Analysis

This section contains exploratory analysis and interactive Streamlit dashboards for individual analysis questions.

#### Running the Streamlit Apps

#### With Virtual Environment
```bash
cd .venv
source ./bin/activate
cd "/workspaces/dsi-ws2025-project-grpab-weigl-mds1ab-awp-proj2/5. Data Analysis/streamlit_apps"
streamlit run industry_insights.py
# or
streamlit run status_analysis.py
```

#### With Requirements file
```bash
pip install -r requirements.txt
cd "/workspaces/dsi-ws2025-project-grpab-weigl-mds1ab-awp-proj2/5. Data Analysis/streamlit_apps"
streamlit run industry_insights.py
# or
streamlit run status_analysis.py
```

---

### 6. Final Application

The final deliverable is a unified Streamlit application that integrates all previous stages into a single coherent interface.

#### Running the final application

#### With Virtual Environment
```bash
cd .venv
source ./bin/activate
cd "/workspaces/dsi-ws2025-project-grpab-weigl-mds1ab-awp-proj2/6. Final Application"
streamlit run app.py
```

#### With Requirements file
```bash
pip install -r requirements.txt
cd "/workspaces/dsi-ws2025-project-grpab-weigl-mds1ab-awp-proj2/6. Final Application"
streamlit run app.py
```
