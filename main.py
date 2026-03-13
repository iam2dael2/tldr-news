from commons.utils.geolocation import get_country_code
from langchain_groq import ChatGroq
from serpapi import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    query: str = input("Input the query you want to search: ")

    params = {
        "engine": "google_news_light",
        "q": "Board of Peace",
        "gl": get_country_code(),
        "api_key": os.getenv("SERP_API_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    news_results = results["news_results"]

    print(news_results)