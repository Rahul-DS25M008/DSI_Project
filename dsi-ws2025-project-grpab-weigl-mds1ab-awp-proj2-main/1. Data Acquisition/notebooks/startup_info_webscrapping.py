from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

# ----------------------------
# European countries list
# ----------------------------
european_countries = [
    "austria","belgium","bulgaria","croatia","cyprus","czech republic","denmark","estonia",
    "finland","france","germany","greece","hungary","ireland","italy","latvia","lithuania",
    "luxembourg","malta","netherlands","poland","portugal","romania","slovakia","slovenia",
    "spain","sweden"
]

# ----------------------------
# Lists to store startup data
# ----------------------------
country_names = []
startup_names = []
startup_descriptions = []
startup_details = []

# ----------------------------
# Start Playwright
# ----------------------------
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # Load main startups page
    page.goto("https://www.failory.com/startups")
    time.sleep(5)  # wait for JavaScript to render content

    html = page.content()
    soup = BeautifulSoup(html, 'html.parser')

    # ----------------------------
    # Find "By Country" section
    # ----------------------------
    by_country_heading = soup.find(
        "h2", 
        class_="main-page-h2", 
        string=lambda text: text and "By Country" in text
    )

    links = []
    if by_country_heading:
        divs = by_country_heading.find_all_next("div", class_="w-dyn-item")
        for div in divs:
            a_tag = div.find("a", href=True)
            if a_tag:
                href = a_tag['href']
                links.append(href)

        # Normalize and filter links for European countries
        links = [
            "/startups/" + link.replace("/startups/", "").lower()
            for link in links
            if link.replace("/startups/", "").lower() in european_countries
        ]

    # ----------------------------
    # Loop through country pages
    # ----------------------------
    for link in links:
        url = f"https://www.failory.com{link}"
        print("Visiting link:", url)
        page.goto(url)
        time.sleep(5)  # wait for page to load

        html = page.content()
        soup = BeautifulSoup(html, 'html.parser')

        content = next(
            (a for a in soup.find_all("article", id="failed-startup-text") 
                if a.find(True, recursive=False) and a.find(True, recursive=False).name == "h3"),
            None
        )
        if not content:
            continue

        h3_tags = content.find_all("h3")
        for h3 in h3_tags:
            country_names.append(link.replace("/startups/", "").capitalize())
            startup_names.append(h3.get_text(strip=True))

            # Extract description
            section_elements = []
            current = h3.find_next_sibling()
            while current and current.name != "h3":
                section_elements.append(current)
                current = current.find_next_sibling()

            description = ""
            figure_seen = False
            for elem in section_elements:
                if elem.name == "figure":
                    figure_seen = True
                elif figure_seen and elem.name == "p":
                    description = elem.get_text(strip=True)
                    break
            startup_descriptions.append(description)

            # Extract details
            details = []
            for elem in section_elements:
                if elem.name == "p" and elem.find("strong") and "Details of the startup:" in elem.get_text():
                    next_ul = elem.find_next_sibling("ul")
                    if next_ul:
                        details = [li.get_text(strip=True) for li in next_ul.find_all("li")]
                    break
            startup_details.append(details)

    browser.close()

# ----------------------------
# Process data and save Excel
# ----------------------------
all_keys = set()
for details in startup_details:
    for item in details:
        if ": " in item:
            key = item.split(": ", 1)[0].strip()
            all_keys.add(key)
all_keys = sorted(all_keys)

data = {
    "Country": country_names,
    "Name": startup_names,
    "Description": startup_descriptions,
}
for key in all_keys:
    data[key] = []

for details in startup_details:
    detail_dict = {}
    for item in details:
        if ": " in item:
            k, v = item.split(": ", 1)
            detail_dict[k.strip()] = v.strip()
    for key in all_keys:
        data[key].append(detail_dict.get(key, ""))

df = pd.DataFrame(data)
if "State" in df.columns:
    df = df.drop(columns=["State"])

# Save CSV inside repo
csv_filename = os.path.join(os.getcwd(), "startups.csv")
df.to_csv(csv_filename, index=False)
print(f"Saved to {csv_filename}")
