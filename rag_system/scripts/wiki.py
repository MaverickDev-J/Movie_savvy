import requests
import json
import cloudscraper
from bs4 import BeautifulSoup

# Serper API function to find Wikipedia link
def search_wikipedia_link(query, api_key):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = json.dumps({
        "q": f"{query} wikipedia",
        "gl": "us",
        "hl": "en",
        "num": 10
    })
    
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        results = response.json()
        
        if "organic" in results:
            for result in results["organic"]:
                if "link" in result and "wikipedia.org" in result["link"]:
                    return result["link"]
        
        return None
    
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return None

# Function to extract table data in markdown format
def extract_table_data(table):
    table_rows = []
    
    tbody = table.find('tbody')
    rows = tbody.find_all('tr') if tbody else table.find_all('tr')
    
    if not rows:
        return "Empty table"
    
    # Process each row
    for row_idx, row in enumerate(rows):
        cells = row.find_all(['th', 'td'])
        if not cells:
            continue
            
        row_data = []
        for cell in cells:
            # Clean cell text
            cell_text = cell.get_text(strip=True, separator=' ')
            # Replace pipe characters to avoid breaking markdown table format
            cell_text = cell_text.replace('|', '\\|') if cell_text else ''
            row_data.append(cell_text)
        
        if row_data:
            # Format as markdown table row
            markdown_row = '| ' + ' | '.join(row_data) + ' |'
            table_rows.append(markdown_row)
            
            # Add separator after header row (first row)
            if row_idx == 0:
                separator = '| ' + ' | '.join(['---'] * len(row_data)) + ' |'
                table_rows.append(separator)
    
    return '\n'.join(table_rows) if table_rows else "Empty table"

# Scrape Wikipedia page content
def scrape_wikipedia_page(url):
    scraper = cloudscraper.create_scraper()
    scraper.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    try:
        response = scraper.get(url)
        response.raise_for_status()
        html_content = response.text
        print("Debug: Successfully retrieved the page.")
    except Exception as e:
        print(f"Debug: Failed to retrieve the page. Error: {e}")
        return {"overview": "Failed to retrieve page", "sections": {}}

    soup = BeautifulSoup(html_content, "html.parser")
    content_area = soup.find("div", class_="mw-content-ltr mw-parser-output")
    if not content_area:
        print("Debug: Content area not found in the HTML.")
        return {"overview": "Content area not found", "sections": {}}

    movie_data = {"overview": "", "sections": {}}
    
    # Collect Overview
    overview_content = []
    elements = content_area.find_all(recursive=False)
    for element in elements:
        if element.name == "div" and "mw-heading" in element.get("class", []):
            break
        if element.name == "p":
            text = element.get_text(strip=True, separator=" ")
            if text:
                overview_content.append(text)
    
    movie_data["overview"] = " ".join(overview_content) if overview_content else "Overview not found"
    
    # Extract all tables
    all_tables = soup.find_all('table')
    print(f"Debug: Found {len(all_tables)} tables on the page")
    
    table_counter = 1
    for table in all_tables:
        table_title = f"Table {table_counter}"
        
        caption = table.find('caption')
        if caption:
            table_title = f"Table: {caption.get_text(strip=True)}"
        else:
            prev_element = table.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if prev_element:
                heading_text = prev_element.get_text(strip=True).replace("[edit]", "").strip()
                if heading_text and len(heading_text) < 100:
                    table_title = f"Table ({heading_text})"
        
        table_content = extract_table_data(table)
        if table_content and table_content != "Empty table":
            movie_data["sections"][table_title] = table_content
            print(f"Debug: Extracted table: {table_title}")
        
        table_counter += 1
    
    # Extract sections under headings
    heading_divs = content_area.find_all("div", class_=lambda x: x and ("mw-heading2" in x or "mw-heading3" in x))
    print(f"Debug: Found {len(heading_divs)} heading sections")
    
    for heading_div in heading_divs:
        heading_tag = heading_div.find(["h2", "h3"])
        if heading_tag:
            section_name = heading_tag.get_text(strip=True).replace("[edit]", "").strip()
            print(f"Debug: Processing section: {section_name}")
            
            section_content = []
            current_element = heading_div.find_next_sibling()
            
            while current_element:
                if (current_element.name == "div" and 
                    current_element.get("class") and
                    any("mw-heading" in cls for cls in current_element.get("class", []))):
                    break
                
                if current_element.name == "p":
                    text = current_element.get_text(strip=True, separator=" ")
                    if text:
                        section_content.append(text)
                elif current_element.name in ["ul", "ol"]:
                    list_items = current_element.find_all("li")
                    for li in list_items:
                        text = li.get_text(strip=True)
                        if text:
                            section_content.append(f"• {text}")
                elif (current_element.name == "div" and 
                      current_element.get("class") and 
                      "div-col" in current_element.get("class", [])):
                    list_items = current_element.find_all("li")
                    for li in list_items:
                        text = li.get_text(strip=True)
                        if text:
                            section_content.append(f"• {text}")
                
                current_element = current_element.find_next_sibling()
            
            if section_content:
                movie_data["sections"][section_name] = " ".join(section_content)
            else:
                movie_data["sections"][section_name] = f"{section_name} content not found"

    return movie_data

def main():
    api_key = ""
    title = input("Enter the search title (e.g., dangal): ")
    
    # First, use Serper API to find Wikipedia link
    wikipedia_link = search_wikipedia_link(title, api_key)
    if wikipedia_link:
        print(f"Found Wikipedia link: {wikipedia_link}")
        movie_data = scrape_wikipedia_page(wikipedia_link)
    else:
        print("No Wikipedia link found via Serper API. Attempting direct scraping with default URL.")
        movie_data = scrape_wikipedia_page("https://en.wikipedia.org/wiki/Steins;Gate_(TV_series)")

    # Save to JSON file
    with open("wikipedia_data.json", "w", encoding="utf-8") as json_file:
        json.dump(movie_data, json_file, ensure_ascii=False, indent=2)
    print("Data saved to wikipedia_data.json")

    # Optional: Print the scraped data
    print("\n" + "="*50)
    print("SCRAPED MOVIE DATA (PREVIEW)")
    print("="*50)
    for section_name, content in movie_data.items():
        if section_name == "overview":
            print(f"\nOVERVIEW:")
            print("-" * 8)
            print(content)
        else:
            print(f"\n{section_name.upper()}:")
            print("-" * len(section_name))
            if "table" in section_name.lower() and "|" in str(content):
                # Display table in formatted way
                print(content)
            elif "cast" in section_name.lower() and "•" in content:
                cast_items = content.split("• ")
                for item in cast_items:
                    if item.strip():
                        print(f"• {item.strip()}")
            else:
                print(content)

    print("\n" + "="*50)
    print(f"Total sections found: {len(movie_data.get('sections', {})) + 1}")
    print("="*50)

if __name__ == "__main__":
    main()
