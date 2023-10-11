import os
from fastapi import APIRouter, HTTPException
from fastapi import File, UploadFile
router = APIRouter()


UPLOAD_DIR = "uploaded_files"
@router.post("/data-otp-csv")
async def upload_data(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="File type not supported")
    
    # Anda bisa memproses file di sini
    # Contoh: baca isi file
    # contents = await file.read()
    
    # Jika Anda memiliki kelas `Preprocess`, Anda bisa memanggilnya di sini:
    # preprocess = Preprocess()
    # preprocess.upload_data(contents)
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    # Menyimpan file ke direktori yang ditentukan
    with open(os.path.join(UPLOAD_DIR, file.filename), "wb") as buffer:
        contents = await file.read()  # baca file
        buffer.write(contents)

    files = []
    if os.path.exists(UPLOAD_DIR):
        files = os.listdir(UPLOAD_DIR)
    return files
