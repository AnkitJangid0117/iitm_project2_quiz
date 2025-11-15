# iitm_project2_quiz

# LLM Quiz Solver – FastAPI (Vercel Deployment)

A fully automated quiz-solving API built using FastAPI and deployed on **Vercel Serverless Functions**.  
This service receives quiz tasks, verifies credentials, loads JavaScript-rendered quiz pages using **Browserless (cloud Playwright)**, solves them, and submits answers within **3 minutes**, as required by the assignment.

---

## 🚀 Features

- ✔ FastAPI backend deployed as a Vercel serverless function  
- ✔ Secret + email verification (HTTP 403 on mismatch)  
- ✔ Automatic JSON validation (HTTP 400 for invalid)  
- ✔ Downloads JavaScript-rendered quiz pages using Browserless Cloud  
- ✔ Extracts question text & parses data  
- ✔ Computes answers programmatically  
- ✔ Submits the answer to the endpoint specified on each quiz page  
- ✔ Handles multi-step quiz chains  
- ✔ Respects the 3-minute time limit  
- ✔ MIT Licensed  
- ✔ Ready for production & evaluation  

---
