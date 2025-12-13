from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import json

# مسیر ذخیره‌سازی فایل‌ها و بازخوردها
UPLOAD_DIR = Path("uploads")
FEEDBACK_FILE = Path("feedback.json")
UPLOAD_DIR.mkdir(exist_ok=True)
if not FEEDBACK_FILE.exists():
    FEEDBACK_FILE.write_text("[]", encoding="utf-8")  # ایجاد فایل خالی JSON برای بازخوردها

# تنظیمات FastAPI و CORS
app = FastAPI(title="SmartProposal Backend")

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "SmartProposal Backend is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    دریافت فایل ODT و ذخیره آن در سرور
    """
    if not file:
        raise HTTPException(status_code=400, detail="هیچ فایلی ارسال نشده است")

    if not file.filename.lower().endswith(".odt"):
        raise HTTPException(status_code=400, detail="فقط فایل با فرمت ODT مجاز است")

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

@app.post("/feedback")
async def submit_feedback(feedback: str):
    """
    ثبت بازخورد کاربر (good, average, bad) و ذخیره آن در فایل JSON
    """
    if feedback not in ["good", "average", "bad"]:
        raise HTTPException(status_code=400, detail="مقدار بازخورد معتبر نیست")

    try:
        # خواندن بازخوردهای قبلی
        existing_feedback = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        # اضافه کردن بازخورد جدید
        existing_feedback.append({"feedback": feedback})
        # ذخیره مجدد در فایل
        FEEDBACK_FILE.write_text(json.dumps(existing_feedback, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "message": "بازخورد شما با موفقیت ثبت شد",
            "feedback": feedback
        }

    except Exception:
        raise HTTPException(status_code=500, detail="خطای ذخیره بازخورد")

@app.get("/feedbacks")
def get_all_feedbacks():
    """
    مشاهده تمام بازخوردهای ثبت شده (برای تست و بررسی)
    """
    try:
        feedbacks = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        return {"feedbacks": feedbacks}
    except Exception:
        raise HTTPException(status_code=500, detail="خطای خواندن بازخوردها")
