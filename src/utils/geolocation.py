import requests

def get_country_code() -> str:
    try:
        code = requests.get("https://ipapi.co/country/", timeout=5).text.strip().lower()
        # Guard against rate-limit error responses (e.g. {"error": true, ...})
        return code if len(code) == 2 and code.isalpha() else "us"
    except Exception:
        return "us"


def get_language_code() -> str:
    """Return the primary BCP-47 language code for the client's locale (e.g. 'id', 'en')."""
    try:
        # ipapi.co/languages/ returns comma-separated codes, e.g. "id,jv,su,ms"
        languages = requests.get("https://ipapi.co/languages/", timeout=5).text.strip()
        primary = languages.split(",")[0]
        # Guard against rate-limit error responses
        return primary if primary.isalpha() else "en"
    except Exception:
        return "en"