import time
import re
import httpx
from bs4 import BeautifulSoup
from solver.browser import fetch_html
from solver.settings import settings

MAX_TIME = 180   # 3 minutes


async def solve_quiz(start_url: str):
    current_url = start_url
    start_time = time.time()

    while True:
        # Timeout check
        if time.time() - start_time > MAX_TIME:
            return {"error": "Time exceeded 3 minutes"}

        # Fetch fully rendered HTML
        html = await fetch_html(current_url)
        soup = BeautifulSoup(html, "html.parser")
        page_text = soup.get_text(" ", strip=True)

        # Very simple answer logic (replace with real parsing)
        answer = compute_answer(page_text)

        # Find submit URL
        submit_url = extract_submit_url(html)
        if not submit_url:
            return {"error": "Submit URL not found"}

        # Submit answer
        payload = {
            "email": settings.STUDENT_EMAIL,
            "secret": settings.STUDENT_SECRET,
            "url": current_url,
            "answer": answer
        }

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(submit_url, json=payload)
            result = r.json()

        # If correct and no further URL → done
        if result.get("correct") and not result.get("url"):
            return {"message": "Quiz complete", "final": True}

        # Move to next quiz
        if "url" in result:
            current_url = result["url"]
            continue

        # If incorrect but no new URL, retry same URL
        continue


def extract_submit_url(html: str) -> str:
    match = re.search(r'https?://[^"]+/submit', html)
    return match.group(0) if match else None


def compute_answer(text: str):
    """
    Simplest placeholder solver logic.
    Replace this with actual parsing rules.
    """
    if "sum" in text.lower():
        return 123  # placeholder
    return "unknown"
