# ==========================================
# FILE: main.py
# ==========================================
# main.py - Enhanced with AIPIPE AI and Base64 Decoding
from dotenv import load_dotenv
import os

# Load .env file (works locally, ignored on Railway)
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import pandas as pd
import PyPDF2
import io
import json
import re
import time
from typing import Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import base64
from urllib.parse import urljoin

app = FastAPI()

# Configuration


MY_SECRET = os.getenv("QUIZ_SECRET", "")
MY_EMAIL = os.getenv("QUIZ_EMAIL", "")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")
AIPIPE_URL = os.getenv("AIPIPE_URL", "")

# Thread pool for running operations
executor = ThreadPoolExecutor(max_workers=3)

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

def extract_text_from_pdf(pdf_content: bytes) -> Dict[str, str]:
    """Extract text from PDF bytes, organized by page"""
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        pages = {}
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            pages[f"page_{page_num}"] = text
        
        return pages
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return {}

def ask_ai(question: str, data: Any, data_type: str = "text") -> Any:
    """Use AIPIPE AI to analyze data and answer questions"""
    if not AIPIPE_TOKEN:
        print("AIPIPE API not configured, falling back to basic analysis")
        return fallback_analysis(question, data)
    
    try:
        # Prepare the data context
        if data_type == "dataframe":
            data_summary = f"""
DataFrame with {len(data)} rows and {len(data.columns)} columns.
Columns: {', '.join(data.columns)}
First 10 rows:
{data.head(10).to_string()}

Summary statistics:
{data.describe().to_string()}

All data (if small enough):
{data.to_string() if len(data) <= 100 else "Dataset too large, showing summary only"}
"""
        elif data_type == "pdf_pages":
            data_summary = "\n\n".join([f"=== {page} ===\n{text}" for page, text in data.items()])
        else:
            data_summary = str(data)[:10000]
        
        prompt = f"""You are helping solve a data analysis quiz. You must provide ONLY the answer value in the correct format.

Question/Task:
{question}

Available Data:
{data_summary}

CRITICAL INSTRUCTIONS - ANSWER FORMAT:
- Provide ONLY the final answer value, absolutely NO explanations or reasoning
- Answer must be ONE of these types:
  * A NUMBER (integer or float): e.g., 12345 or 123.45
  * A BOOLEAN: true or false
  * A STRING: e.g., New York (without quotes)
  * A BASE64 URI: e.g., data:image/png;base64,iVBORw0KG...
- DO NOT return JSON objects or arrays
- If the question asks for multiple values, return just the primary answer
- For calculations, return ONLY the final number
- Do not include units, currency symbols, or explanations

Answer:"""

        headers = {
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "input": prompt
        }
        
        print(f"Calling AIPIPE API...")
        response = requests.post(AIPIPE_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # Parse AIPIPE response format
        try:
            answer_text = result.get("output", [{}])[0].get("content", [{}])[0].get("text", "")
        except (KeyError, IndexError, TypeError):
            answer_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        answer_text = answer_text.strip()
        print(f"AI answer: {answer_text}")
        
        # Clean up markdown/code blocks
        answer_text = re.sub(r'^```json\s*|\s*```$', '', answer_text, flags=re.MULTILINE)
        answer_text = answer_text.strip('`').strip('"').strip("'").strip()
        
        # Try to parse as number first
        try:
            if re.match(r'^-?\d+\.?\d*$', answer_text):
                if '.' in answer_text:
                    return float(answer_text)
                return int(answer_text)
        except:
            pass
        
        # Check for boolean
        if answer_text.lower() in ['true', 'false']:
            return answer_text.lower() == 'true'
        
        # Check if it's a base64 string
        if re.match(r'^data:[^;]+;base64,[A-Za-z0-9+/]+=*$', answer_text):
            return answer_text
        
        return answer_text
        
    except Exception as e:
        print(f"Error asking AI: {e}")
        import traceback
        traceback.print_exc()
        return fallback_analysis(question, data)

def fallback_analysis(question: str, data: Any) -> Any:
    """Fallback analysis without AI"""
    question_lower = question.lower()
    
    if isinstance(data, pd.DataFrame):
        print(f"DataFrame shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        
        if 'sum' in question_lower:
            for col in data.columns:
                if col.lower() in question_lower or 'value' in col.lower():
                    try:
                        return int(data[col].sum())
                    except:
                        pass
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                return int(data[numeric_cols[0]].sum())
        
        if 'count' in question_lower or 'how many' in question_lower:
            return len(data)
        
        if 'average' in question_lower or 'mean' in question_lower:
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                return float(data[numeric_cols[0]].mean())
        
        if 'max' in question_lower or 'highest' in question_lower:
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                return int(data[numeric_cols[0]].max())
        
        if 'min' in question_lower or 'lowest' in question_lower:
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                return int(data[numeric_cols[0]].min())
    
    if isinstance(data, (str, dict)):
        data_str = str(data)
        numbers = [float(x) for x in re.findall(r'-?\d+\.?\d*', data_str)]
        if numbers:
            if 'sum' in question_lower:
                return int(sum(numbers))
            if 'count' in question_lower:
                return len(numbers)
    
    return 0

def download_and_process_file(file_url: str, question: str) -> Any:
    """Download and process file with AI assistance"""
    print(f"Downloading file: {file_url}")
    
    response = requests.get(file_url, timeout=30)
    response.raise_for_status()
    
    content_type = response.headers.get('content-type', '')
    
    if 'pdf' in content_type.lower() or file_url.endswith('.pdf'):
        pages = extract_text_from_pdf(response.content)
        print(f"Extracted PDF with {len(pages)} pages")
        return ask_ai(question, pages, "pdf_pages")
    
    if 'csv' in content_type.lower() or file_url.endswith('.csv'):
        df = pd.read_csv(io.StringIO(response.text))
        print(f"Loaded CSV with {len(df)} rows, {len(df.columns)} columns")
        return ask_ai(question, df, "dataframe")
    
    if 'excel' in content_type.lower() or 'spreadsheet' in content_type.lower() or file_url.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(io.BytesIO(response.content))
        print(f"Loaded Excel with {len(df)} rows, {len(df.columns)} columns")
        return ask_ai(question, df, "dataframe")
    
    if 'json' in content_type.lower() or file_url.endswith('.json'):
        data = json.loads(response.text)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            return ask_ai(question, df, "dataframe")
        return ask_ai(question, data, "json")
    
    return ask_ai(question, response.text, "text")

def solve_quiz(quiz_url: str) -> Dict[str, Any]:
    """Solve a single quiz using HTTP requests"""
    print(f"Solving quiz: {quiz_url}")
    
    # Fetch the page
    response = requests.get(quiz_url, timeout=30)
    response.raise_for_status()
    
    # Parse HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Check for base64 encoded content in JavaScript
    page_text = None
    
    # Look for atob() JavaScript pattern (base64 decoding)
    script_tags = soup.find_all('script')
    for script in script_tags:
        script_content = script.string
        if script_content and 'atob' in script_content:
            # Extract base64 content from atob(`...`)
            match = re.search(r'atob\([`\'"]([^`\'"]+)[`\'"]', script_content)
            if match:
                try:
                    base64_content = match.group(1)
                    decoded = base64.b64decode(base64_content).decode('utf-8')
                    page_text = decoded
                    print("✓ Decoded base64 JavaScript content")
                    break
                except Exception as e:
                    print(f"Failed to decode base64: {e}")
    
    # Fallback to regular text extraction
    if not page_text:
        page_text = soup.get_text()
        print("Using plain text content")
    
    print(f"Page content (first 500 chars): {page_text[:500]}")
    
    # Extract submit URL - handle both absolute and relative URLs
    submit_url = None
    
    # Try to find full URL first
    submit_url_match = re.search(r'https?://[^\s]+/submit', page_text)
    if submit_url_match:
        submit_url = submit_url_match.group(0)
    else:
        # Look for relative URL pattern like "to /submit" or "POST to /submit"
        relative_match = re.search(r'to\s+(/[^\s]+/submit)', page_text)
        if relative_match:
            relative_path = relative_match.group(1)
            submit_url = urljoin(quiz_url, relative_path)
        else:
            # Try just looking for "/submit"
            if '/submit' in page_text:
                submit_url = urljoin(quiz_url, '/submit')
            else:
                raise Exception("Could not find submit URL in page")
    
    print(f"Submit URL: {submit_url}")
    
    # Find file links
    file_url = None
    links = soup.find_all('a', href=True)
    for link in links:
        href = link['href']
        text = link.get_text().lower()
        if re.search(r'\.(pdf|csv|xlsx?|json|txt)$', href, re.I) or 'file' in text or 'download' in text:
            # Handle relative file URLs
            if not href.startswith('http'):
                file_url = urljoin(quiz_url, href)
            else:
                file_url = href
            break
    
    # Also check in decoded page_text for file URLs
    if not file_url:
        file_match = re.search(r'href=["\']([^"\']+\.(pdf|csv|xlsx?|json|txt))["\']', page_text, re.I)
        if file_match:
            file_url = urljoin(quiz_url, file_match.group(1))
    
    # Solve the question
    answer = None
    
    if file_url:
        print(f"Found file URL: {file_url}")
        answer = download_and_process_file(file_url, page_text)
    else:
        # Try API endpoint
        api_match = re.search(r'https?://[^\s]+/api[^\s]*', page_text)
        if api_match:
            api_url = api_match.group(0)
            print(f"Found API URL: {api_url}")
            api_response = requests.get(api_url, timeout=30)
            api_data = api_response.json() if 'json' in api_response.headers.get('content-type', '') else api_response.text
            answer = ask_ai(page_text, api_data, "json" if isinstance(api_data, (dict, list)) else "text")
        else:
            answer = ask_ai(page_text, page_text, "text")
    
    print(f"Calculated answer: {answer}")
    
    return {
        "answer": answer,
        "submit_url": submit_url
    }

def submit_answer(submit_url: str, quiz_url: str, answer: Any) -> Dict[str, Any]:
    """Submit answer to the quiz endpoint"""
    payload = {
        "email": MY_EMAIL,
        "secret": MY_SECRET,
        "url": quiz_url,
        "answer": answer
    }
    
    print(f"Submitting to {submit_url}: {payload}")
    
    response = requests.post(submit_url, json=payload, timeout=30)
    response.raise_for_status()
    
    return response.json()

def process_quiz_chain(initial_url: str):
    """Process a chain of quizzes"""
    current_url = initial_url
    max_quizzes = 10
    quiz_count = 0
    start_time = time.time()
    
    while current_url and quiz_count < max_quizzes:
        quiz_count += 1
        elapsed = time.time() - start_time
        print(f"\n=== Quiz {quiz_count}: {current_url} ===")
        print(f"Elapsed time: {elapsed:.1f}s")
        
        if elapsed > 170:
            print("⚠️ Approaching 3-minute timeout, stopping")
            break
        
        try:
            result = solve_quiz(current_url)
            submit_response = submit_answer(result["submit_url"], current_url, result["answer"])
            
            print(f"Submit response: {submit_response}")
            
            if submit_response.get("correct"):
                print("✓ Answer correct!")
                current_url = submit_response.get("url")
                if not current_url:
                    print("🎉 Quiz chain completed successfully!")
                    break
            else:
                reason = submit_response.get("reason", "Unknown error")
                print(f"✗ Answer incorrect: {reason}")
                next_url = submit_response.get("url")
                if next_url and next_url != current_url:
                    print(f"Moving to next quiz: {next_url}")
                    current_url = next_url
                else:
                    break
                    
        except Exception as e:
            print(f"❌ Error in quiz {quiz_count}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Processed {quiz_count} quizzes in {total_time:.1f} seconds")
    print(f"{'='*50}")

@app.post("/quiz")
async def handle_quiz(request: QuizRequest):
    """Main endpoint to receive quiz requests"""
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, process_quiz_chain, request.url)
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "accepted",
            "message": "Quiz processing started"
        }
    )

@app.get("/")
async def root():
    return {
        "message": "Quiz API Endpoint with AI (Base64 Decoding Support)", 
        "status": "running",
        "ai_enabled": bool(AIPIPE_TOKEN)
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ai_enabled": bool(AIPIPE_TOKEN)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


