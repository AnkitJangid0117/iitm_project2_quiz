# LLM Analysis Quiz Solver (FastAPI + AIPIPE AI)

## 🚀 Project Overview

This project automates solving data-analysis quizzes hosted online. It:

* Downloads quiz pages
* Decodes Base64 hidden content
* Processes PDF/CSV/XLSX/JSON/TXT files
* Uses AI to compute answers
* Submits responses
* Automatically chains through multiple quizzes

## 📁 Features

* PDF text extraction using PyPDF2
* CSV/Excel/JSON auto-loading via pandas
* Base64 JavaScript decoding
* AIPIPE AI integration with strict output rules
* Fallback analysis engine
* Quiz chaining (up to 10 quizzes)

## 🔧 API Endpoints

### **POST /quiz**

Starts quiz solving.

```
{
  "email": "23f1001630@ds.study.iitm.ac.in",
  "secret": "my_secret",
  "url": "https://tds-llm-analysis.s-anand.net/demo"
}
```

Response:

```
{"status": "accepted", "message": "Quiz processing started"}
```

### **GET /**

Returns basic status.

### **GET /health**

Health check.

## 🔑 Environment Variables (.env)

```
QUIZ_SECRET=your_secret
QUIZ_EMAIL=23f1001630@ds.study.iitm.ac.in
AIPIPE_TOKEN=your_aipipe_token
AIPIPE_URL=AIPIPE_URL
PORT=port
```

## ▶️ Running Locally

```
pip install -r requirements.txt
uvicorn main:app --reload
```

## ☁️ Deploying to Railway

1. Create project
2. Add environment vars
3. Deploy

## 🧠 AIPIPE Answering Rules

* Only final answer
* No reasoning
* No JSON structure
* No units or quotes
* Supports Base64 image URIs

## 📝 CURL Example

```
curl -X POST https://your-api/quiz -H "Content-Type: application/json" -d '{"email":"23f1001630@ds.study.iitm.ac.in","secret":"my_secret","url":"https://tds-llm-analysis.s-anand.net/demo"}'
```

## 📌 Required Email Note

Email included: **[23f1001630@ds.study.iitm.ac.in](mailto:23f1001630@ds.study.iitm.ac.in)**
