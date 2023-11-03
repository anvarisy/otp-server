import os
from fastapi import APIRouter, HTTPException
from fastapi import File, UploadFile

from sender.dto import UploadResponse
router = APIRouter()


UPLOAD_DIR = "uploaded_files"
@router.post("/data-otp-csv", response_model=UploadResponse)
async def upload_data(file: UploadFile = File(...)):
        try:
            if file.content_type != "text/csv":
                raise HTTPException(status_code=400, detail="File type not supported")
            if not os.path.exists(UPLOAD_DIR):
                os.makedirs(UPLOAD_DIR)

            # Menyimpan file ke direktori yang ditentukan
            with open(os.path.join(UPLOAD_DIR, file.filename), "wb") as buffer:
                contents = await file.read()  # baca file
                buffer.write(contents)

            files = []
            if os.path.exists(UPLOAD_DIR):
                files = os.listdir(UPLOAD_DIR)
            
            return {
                "error": "",
                "status": True,
                "data": files
            }
    
        except HTTPException as e:
            return {
                "error": e.detail,
                "status": False,
                "data": []
            }
