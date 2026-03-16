import trafilatura

def fetch_article_content(url: str) -> str:
    """Extract full article text from a URL using trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            return text or ""
    except Exception:
        return ""
    return ""

if __name__ == "__main__":
    url: str = "https://id.wikipedia.org/wiki/Jokowi"
    content: str = fetch_article_content(url)

    print(content)