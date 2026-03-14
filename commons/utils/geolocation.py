import requests

def get_country_code() -> str:
    try:
        return requests.get("https://ipapi.co/country/", timeout=5).text.strip().lower()
    except Exception:
        return "us"


def get_language_code() -> str:
    """Return the primary BCP-47 language code for the client's locale (e.g. 'id', 'en')."""
    try:
        # ipapi.co/languages/ returns comma-separated codes, e.g. "id,jv,su,ms"
        languages = requests.get("https://ipapi.co/languages/", timeout=5).text.strip()
        primary = languages.split(",")[0]
        return primary if primary else "en"
    except Exception:
        return "en"