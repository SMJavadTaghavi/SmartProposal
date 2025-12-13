from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil


origins = [
    "http://localhost:3000"
]

app = FastAPI(title="SmartProposal Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Receives an ODT file and saves it to server
    Handles server and validation errors properly
    """


    if not file:
        raise HTTPException(
            status_code=400,
            detail="هیچ فایلی ارسال نشده است"
        )


    if not file.filename.lower().endswith(".odt"):
        raise HTTPException(
            status_code=400,
            detail="فقط فایل با فرمت ODT مجاز است"
        )

    try:
        save_path = UPLOAD_DIR / file.filename


        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "message": "فایل با موفقیت آپلود شد"
            }
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="خطای داخلی سرور. لطفاً بعداً دوباره تلاش کنید"
        )



@app.get("/")
def root():
    return {"message": "SmartProposal Backend is running"}

@app.post("/feedback")
async def submit_feedback(feedback: str):
    """
    Collects lightweight user feedback for UI experience evaluation
    """

    if feedback not in ["good", "average", "bad"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid feedback value"
        )

    # فعلاً فقط لاگ می‌کنیم (برای تست UX کافیه)
    return {
        "message": "بازخورد شما با موفقیت ثبت شد",
        "feedback": feedback
    }
