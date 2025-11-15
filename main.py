# main.py - Enhanced with AIPIPE AI and Playwright
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
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
import os

app = FastAPI()

# Configuration
MY_SECRET = os.getenv("QUIZ_SECRET", "")
MY_EMAIL = os.getenv("QUIZ_EMAIL", "")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")
AIPIPE_URL = os.getenv("AIPIPE_URL", "")

# Thread pool for running sync browser operations
executor = ThreadPoolExecutor(max_workers=3)

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

class QuizResponse(BaseModel):
    status: str
    message: Optional[str] = None

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
            data_summary = str(data)[:10000]  # Limit context size
        
        # Ask AI to analyze
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
            # Fallback to OpenAI format
            answer_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        answer_text = answer_text.strip()
        print(f"AI answer: {answer_text}")
        
        # Clean up markdown/code blocks
        answer_text = re.sub(r'^```json\s*|\s*```$', '', answer_text, flags=re.MULTILINE)
        answer_text = answer_text.strip('`').strip('"').strip("'").strip()
        
        # Try to parse as number first (most common for data analysis)
        try:
            # Check if it looks like a number
            if re.match(r'^-?\d+\.?\d*$', answer_text):
                if '.' in answer_text:
                    return float(answer_text)
                return int(answer_text)
        except:
            pass
        
        # Check for boolean
        if answer_text.lower() in ['true', 'false']:
            return answer_text.lower() == 'true'
        
        # Check if it's a base64 string (for file attachments)
        if re.match(r'^data:[^;]+;base64,[A-Za-z0-9+/]+=*$', answer_text):
            return answer_text
        
        # Return as string (don't return JSON objects)
        return answer_text
        
    except Exception as e:
        print(f"Error asking AI: {e}")
        import traceback
        traceback.print_exc()
        return fallback_analysis(question, data)

def fallback_analysis(question: str, data: Any) -> Any:
    """Fallback analysis without AI"""
    question_lower = question.lower()
    
    # If data is a DataFrame
    if isinstance(data, pd.DataFrame):
        print(f"DataFrame shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        
        # Sum of a column
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
        
        # Count rows
        if 'count' in question_lower or 'how many' in question_lower:
            return len(data)
        
        # Average
        if 'average' in question_lower or 'mean' in question_lower:
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                return float(data[numeric_cols[0]].mean())
        
        # Maximum
        if 'max' in question_lower or 'highest' in question_lower:
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                return int(data[numeric_cols[0]].max())
        
        # Minimum
        if 'min' in question_lower or 'lowest' in question_lower:
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                return int(data[numeric_cols[0]].min())
    
    # If data is text or dict
    if isinstance(data, (str, dict)):
        data_str = str(data)
        numbers = [float(x) for x in re.findall(r'-?\d+\.?\d*', data_str)]
        if numbers:
            if 'sum' in question_lower:
                return int(sum(numbers))
            if 'count' in question_lower:
                return len(numbers)
            if 'average' in question_lower:
                return sum(numbers) / len(numbers)
    
    return 0

def download_and_process_file(file_url: str, question: str) -> Any:
    """Download and process file with AI assistance"""
    print(f"Downloading file: {file_url}")
    
    response = requests.get(file_url, timeout=30)
    response.raise_for_status()
    
    content_type = response.headers.get('content-type', '')
    
    # Handle PDF
    if 'pdf' in content_type.lower() or file_url.endswith('.pdf'):
        pages = extract_text_from_pdf(response.content)
        print(f"Extracted PDF with {len(pages)} pages")
        return ask_ai(question, pages, "pdf_pages")
    
    # Handle CSV
    if 'csv' in content_type.lower() or file_url.endswith('.csv'):
        df = pd.read_csv(io.StringIO(response.text))
        print(f"Loaded CSV with {len(df)} rows, {len(df.columns)} columns")
        return ask_ai(question, df, "dataframe")
    
    # Handle Excel
    if 'excel' in content_type.lower() or 'spreadsheet' in content_type.lower() or file_url.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(io.BytesIO(response.content))
        print(f"Loaded Excel with {len(df)} rows, {len(df.columns)} columns")
        return ask_ai(question, df, "dataframe")
    
    # Handle JSON
    if 'json' in content_type.lower() or file_url.endswith('.json'):
        data = json.loads(response.text)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            return ask_ai(question, df, "dataframe")
        return ask_ai(question, data, "json")
    
    # Default: treat as text
    return ask_ai(question, response.text, "text")

def solve_quiz(quiz_url: str) -> Dict[str, Any]:
    """Solve a single quiz using Playwright"""
    print(f"Solving quiz: {quiz_url}")
    
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu'
            ]
        )
        
        try:
            page = browser.new_page()
            page.goto(quiz_url, wait_until='networkidle', timeout=30000)
            
            # Wait a bit for JS to execute
            time.sleep(2)
            
            # Get page content
            page_text = page.inner_text('body')
            print(f"Page content (first 500 chars): {page_text[:500]}")
            
            # Extract submit URL
            submit_url_match = re.search(r'https?://[^\s]+/submit', page_text)
            if not submit_url_match:
                raise Exception("Could not find submit URL in page")
            
            submit_url = submit_url_match.group(0)
            print(f"Submit URL: {submit_url}")
            
            # Find file links
            file_url = None
            links = page.query_selector_all('a[href]')
            
            for link in links:
                href = link.get_attribute('href')
                text = link.inner_text().lower()
                if href and (re.search(r'\.(pdf|csv|xlsx?|json|txt)$', href, re.I) or 
                            'file' in text or 'download' in text):
                    file_url = href
                    break
            
            # Solve the question with AI
            answer = None
            
            if file_url:
                print(f"Found file URL: {file_url}")
                answer = download_and_process_file(file_url, page_text)
            else:
                # Try to extract API endpoint
                api_match = re.search(r'https?://[^\s]+/api[^\s]*', page_text)
                if api_match:
                    api_url = api_match.group(0)
                    print(f"Found API URL: {api_url}")
                    api_response = requests.get(api_url, timeout=30)
                    api_data = api_response.json() if 'json' in api_response.headers.get('content-type', '') else api_response.text
                    answer = ask_ai(page_text, api_data, "json" if isinstance(api_data, (dict, list)) else "text")
                else:
                    # Use AI to understand the question from page content
                    answer = ask_ai(page_text, page_text, "text")
            
            print(f"Calculated answer: {answer}")
            
            return {
                "answer": answer,
                "submit_url": submit_url
            }
            
        finally:
            browser.close()

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
        
        # Check 3-minute timeout
        if elapsed > 170:  # 170 seconds = 2min 50sec, leave buffer
            print("⚠️ Approaching 3-minute timeout, stopping")
            break
        
        try:
            # Solve the quiz
            result = solve_quiz(current_url)
            
            # Submit the answer
            submit_response = submit_answer(
                result["submit_url"],
                current_url,
                result["answer"]
            )
            
            print(f"Submit response: {submit_response}")
            
            # Check if correct and get next URL
            if submit_response.get("correct"):
                print("✓ Answer correct!")
                current_url = submit_response.get("url")
                if not current_url:
                    print("🎉 Quiz chain completed successfully!")
                    break
            else:
                reason = submit_response.get("reason", "Unknown error")
                print(f"✗ Answer incorrect: {reason}")
                # Try next URL if provided
                next_url = submit_response.get("url")
                if next_url and next_url != current_url:
                    print(f"Moving to next quiz: {next_url}")
                    current_url = next_url
                else:
                    print("No more quizzes to attempt")
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
    
    # Validate secret
    if request.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    # Process quiz in background
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
        "message": "Quiz API Endpoint with AIPIPE AI (Playwright)", 
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
    uvicorn.run(app, host="0.0.0.0", port=8000)