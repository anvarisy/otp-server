from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI
from dbconf import conf
from sqlalchemy import MetaData
from fastapi import HTTPException
import logging
from routers import otp_router, file_router, ml_router
from uvicorn import run
import uvicorn
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

app = FastAPI()
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="OTP Classification Service",
        version="1.0.0",
        description="This is for education project",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema
metadata = MetaData()
app.openapi = custom_openapi


# app.include_router(otp_router.router, prefix="/otp", tags=["OTP"])
app.include_router(file_router.router, prefix="/upload", tags=["Upload"])
app.include_router(ml_router.router, prefix="/ml", tags=["Machine Learning"])

@app.on_event("startup")
async def startup():
    try:
        await conf.database.connect()
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail="Database connection error")
@app.on_event("shutdown")
async def shutdown():
    await conf.database.disconnect()

# @app.get("/merchant/seed")
# async def seed_merchant():
#     query = "INSERT INTO tb_merchant(merchant_name) VALUES (:merchant_name)"
#     values = {"merchant_name": "Fazpas"}
#     await database.execute(query, values)
#     return {"status": "success"}

# @app.get("/provider/seed")
# async def seed_merchant():
#     query = "INSERT INTO tb_provider(provider_name, price) VALUES (:provider_name, :price)"
#     values = {"provider_name": "Indosat", "price": 500}
#     await database.execute(query, values)
#     return {"status": "success"}


# @app.get("/files/", response_model=List[str])
# async def list_files():
#     """
#     List all uploaded files.
#     """
#     files = []
#     if os.path.exists(UPLOAD_DIR):
#         files = os.listdir(UPLOAD_DIR)
#     return files

# @app.get("/files/{filename}")
# async def read_file(filename: str):
#     """
#     Read the content of a specific file.
#     """
#     file_path = os.path.join(UPLOAD_DIR, filename)
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="File not found")

#     with open(file_path, "r") as f:
#         content = f.read()

#     return {"filename": filename, "content": content}
  # Gantikan 'nama_modul' dengan nama modul Anda

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=2000, reload=True)
