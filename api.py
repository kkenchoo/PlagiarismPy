from fastapi import FastAPI
from pydantic import BaseModel
from plagiarismdetection import PythonPlagiarismDetector

app = FastAPI()

class CodeInput(BaseModel):
    code1: str
    code2: str

detector = PythonPlagiarismDetector()

@app.post("/check_plagiarism/")
async def check_plagiarism(input_data: CodeInput):
    code1 = input_data.code1
    code2 = input_data.code2

    print("Received code1:", code1)
    print("Received code2:", code2)

    result = detector.compare_codes(code1, code2)

    print("Plagiarism detection result:", result)

    return result
