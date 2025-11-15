import httpx
from solver.settings import settings


async def fetch_html(url: str) -> str:
    """
    Render JavaScript page using Browserless Cloud (Playwright API).
    Works on Vercel serverless.
    """
    endpoint = f"https://chrome.browserless.io/playwright?token={settings.BROWSERLESS_KEY}"

    payload = {
        "url": url,
        "waitUntil": "networkidle"
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(endpoint, json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("html", "")
