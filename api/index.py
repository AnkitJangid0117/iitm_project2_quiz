from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mangum import Mangum
from pydantic import BaseModel
from solver.logic import solve_quiz
from solver.settings import settings

app = FastAPI()
handler = Mangum(app)


class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


@app.post("/")
async def entry(req: QuizRequest):
    # Validate secret
    if req.secret != settings.STUDENT_SECRET:
        return JSONResponse({"detail": "Invalid secret"}, status_code=403)

    # Validate email
    if req.email != settings.STUDENT_EMAIL:
        return JSONResponse({"detail": "Invalid email"}, status_code=403)

    result = await solve_quiz(req.url)
    return {"status": "ok", "result": result}
