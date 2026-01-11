import pandas as pd
import numpy as np
import re

df = pd.read_csv("1. Data Acquisition/data/startups.csv")

# --------------------------------------------------------
# Clean the "Name" column (remove numbering like "1) ")
# --------------------------------------------------------
df["Name"] = df["Name"].str.replace(r"^\s*\d+\)\s*", "", regex=True)

# --------------------------------------------------------
# Extract Currency Column
# --------------------------------------------------------
def extract_currency_symbol(value):
    if isinstance(value, str):
        if value.startswith("$"): return "USD"
        if value.startswith("€"): return "EUR"
        if value.startswith("£"): return "GBP"
        if value.startswith("RON"): return "RON"
        if value.startswith("SEK"): return "SEK"
        if value.startswith("DKK"): return "DKK"
        if value.startswith("PLN"): return "PLN"
        if value.startswith("RUB"): return "RUB"
    return None

df["Currency"] = df["Funding"].apply(extract_currency_symbol)

# --------------------------------------------------------
# Clean Funding 
# --------------------------------------------------------
df["Funding"] = (
    df["Funding"]
    .str.replace(r"[^\d.]", "", regex=True)
    .astype(float)
)

# --------------------------------------------------------
# Convert all Fundings into EURO 
# --------------------------------------------------------

# define conversion rates
rates = {
    "USD": 0.8584,
    "GBP": 1.1316,
    "EUR": 1.0,
    "RON": 0.1965,
    "SEK": 0.09070,
    "DKK": 0.1339,
    "PLN": 0.2359,
    "RUB": 0.01069
}

df["Funding_EUR"] = df.apply(
    lambda row: row["Funding"] * rates.get(row["Currency"], 1.0),
    axis=1
)

# --------------------------------------------------------
# Clean employee range
# --------------------------------------------------------
def clean_employee_range(x):
    if isinstance(x, str) and "-" in x:
        low, high = x.split("-")
        return (int(low) + int(high)) / 2
    return pd.to_numeric(x, errors="coerce")

df["Number of employees"] = df["Number of employees"].apply(clean_employee_range)

# --------------------------------------------------------
# Clean Investor  
# --------------------------------------------------------

# Save the raw string sothat we can check the text
df["Investors_raw"] = df["Number of investors"]

# Extract investor count
df["Number of investors"] = (
    df["Investors_raw"]
    .str.extract(r"(\d+)")
    .astype("Int64")     
)

# Extract investor names 
def parse_investors(raw):
    if not isinstance(raw, str):
        return []
    match = re.search(r"\((.*?)\)", raw)
    if not match:
        return []
    investors_str = match.group(1)
    return [i.strip() for i in investors_str.split(",")]

df["Investors"] = df["Investors_raw"].apply(parse_investors)

# --------------------------------------------------------
# Founders & Industries & Investors
# --------------------------------------------------------
def safe_split_list(value):
    if isinstance(value, str):
        return [v.strip() for v in value.split(",")]
    return []  # if NaN or bad data

df["Founders"] = df["Founders"].apply(safe_split_list)
df["Industries"] = df["Industries"].apply(safe_split_list)

# save elements in separate rows
df = df.explode("Founders")
df = df.explode("Industries")
df = df.explode("Investors")

# --------------------------------------------------------
# Clean remaining whitespace
# --------------------------------------------------------
for col in ["Country", "Name", "Description", "City", "Founders", "Industries", "Investors"]:
    df[col] = df[col].astype(str).str.strip()

# --------------------------------------------------------
# Convert Started in integer
# --------------------------------------------------------
df["Started in"] = pd.to_numeric(df["Started in"], errors="coerce")

# --------------------------------------------------------
# Save final dataset
# --------------------------------------------------------

# Remove Investors_raw column
df = df.drop(columns=["Investors_raw"])
df.to_csv("2. Data Cleaning/data/startups_info_websp_clean.csv", index=False)

print("Cleaning complete! Saved as 2. Data Cleaning/data/startups_info_websp_clean.csv")
print(df.head(20))
