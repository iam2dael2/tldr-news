from pathlib import Path
import requests

CLIENT_COUNTRY_API: Path = Path(r"https://ipapi.co/country/")

def get_country_code():
    try:
        response = requests.get(CLIENT_COUNTRY_API)
        return response.text.strip().lower()  # return "id", "us", "gb", dst

    except:
        return "us"  # fallback default