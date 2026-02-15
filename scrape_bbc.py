import requests
import os
import feedparser
from bs4 import BeautifulSoup
from newspaper import Article
from time import sleep

HEADERS = {"User-Agent": "Mozilla/5.0"}
SAVE_DIR = "sports_politics_data"
MAX_ARTICLES = 300


# =====================================================
# Utility: Extract text using BeautifulSoup (BBC)
# =====================================================
def extract_bbc_article(url):
    response = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return " ".join(p.get_text() for p in paragraphs).strip()


# =====================================================
# Utility: Extract text using Newspaper (Guardian RSS)
# =====================================================
def extract_guardian_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text.strip()


# =====================================================
# Get BBC links from category pages
# =====================================================
def get_bbc_links(page_url, keyword):
    response = requests.get(page_url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if keyword in href and href.startswith("/"):
            links.add("https://www.bbc.com" + href)

    return links


# =====================================================
# Scrape BBC category (Sports or Politics)
# =====================================================
def scrape_bbc_category(category, pages, keyword):
    os.makedirs(f"{SAVE_DIR}/{category}", exist_ok=True)

    existing_files = len(os.listdir(f"{SAVE_DIR}/{category}"))
    count = existing_files

    all_links = set()
    for page in pages:
        print(f"[BBC] Collecting links from {page}")
        all_links |= get_bbc_links(page, keyword)

    print(f"[BBC] Total unique links for {category}: {len(all_links)}")

    for link in all_links:
        if count >= MAX_ARTICLES:
            break

        try:
            text = extract_bbc_article(link)

            if len(text.split()) < 150:
                continue

            count += 1
            with open(f"{SAVE_DIR}/{category}/{category}_{count:03}.txt",
                      "w", encoding="utf-8") as f:
                f.write(text)

            print(f"[BBC-{category.upper()}] Saved {count}")
            sleep(1)

        except Exception:
            continue


# =====================================================
# Scrape Guardian RSS (Politics Only)
# =====================================================
def scrape_guardian_politics():
    RSS_FEEDS = [
        "https://www.theguardian.com/politics/rss",
        "https://www.theguardian.com/world/rss"
    ]

    category = "politics"
    os.makedirs(f"{SAVE_DIR}/{category}", exist_ok=True)

    existing_files = len(os.listdir(f"{SAVE_DIR}/{category}"))
    count = existing_files

    links = set()
    for feed in RSS_FEEDS:
        feed_data = feedparser.parse(feed)
        for entry in feed_data.entries:
            links.add(entry.link)

    print(f"[Guardian] Collected {len(links)} RSS links")

    for link in links:
        if count >= MAX_ARTICLES:
            break

        try:
            text = extract_guardian_article(link)

            if len(text.split()) < 150:
                continue

            count += 1
            with open(f"{SAVE_DIR}/{category}/{category}_{count:03}.txt",
                      "w", encoding="utf-8") as f:
                f.write(text)

            print(f"[Guardian-POLITICS] Saved {count}")
            sleep(1)

        except Exception:
            continue


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    sports_pages = [
        "https://www.bbc.com/sport",
        "https://www.bbc.com/sport/football",
        "https://www.bbc.com/sport/cricket",
        "https://www.bbc.com/sport/tennis",
        "https://www.bbc.com/sport/formula1"
    ]

    politics_pages = [
        "https://www.bbc.com/news/politics",
        "https://www.bbc.com/news/uk-politics",
        "https://www.bbc.com/news/world-politics",
        "https://www.bbc.com/news/elections"
    ]

    # BBC Sports
    scrape_bbc_category("sports", sports_pages, "/sport/")

    # BBC Politics
    scrape_bbc_category("politics", politics_pages, "/news/")

    # Guardian Politics (adds more politics articles)
    scrape_guardian_politics()

    print("\n✅ Data collection complete.")