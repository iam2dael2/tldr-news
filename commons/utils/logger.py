import logging
import os

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("tldr_news")
logger.setLevel(logging.DEBUG)

# File handler — full DEBUG detail written to logs/tldr_news.log
file_handler = logging.FileHandler("logs/tldr_news.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))

# Console handler — WARNING and above only (errors surface to terminal)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
