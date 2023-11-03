import os
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

from ml.random_forest import RandomForestImplementation

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def view_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "name": " & Welcome Back !"})

@router.get("/upload", response_class=HTMLResponse)
async def view_home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


UPLOAD_DIR = "uploaded_files"
@router.get("/train", response_class=HTMLResponse)
async def view_home(request: Request):
    files = None
    if os.path.exists(UPLOAD_DIR):
        files = os.listdir(UPLOAD_DIR)
    return templates.TemplateResponse("train.html", {"request": request, "files":files})

@router.post("/train-result", response_class=HTMLResponse)
async def train_result(
    request: Request,
    selectedFile: str = Form(...),
    treesCount: int = Form(...),
    splitPercentage: float = Form(...)
):
    rf = RandomForestImplementation()
    filename = f'./uploaded_files/{selectedFile}'
    df = pd.read_csv(filename)
    try:
        result = rf.train(df, treesCount, splitPercentage)
        return templates.TemplateResponse("results.html", 
            {"request": request, "request": request,
            "metrics": result['metrics'],
            "plot_url": f"/static/{result['plot_filename']}",
            "voting_results": result['voting_results'],
            "tree_count": treesCount,
            "tree_images": [{'url': f"/static/trees/{filename}", 'num': i+1} for i, filename in enumerate(result['tree_filenames'])]
            })
    except Exception as e:
        print(f"An error occurred: {e}")
        return templates.TemplateResponse("results.html", {"request": request, 'metrics': {}, 'plot_filename': '', 'voting_results': [], "tree_image_urls":[]})
    

TREE_DIR = "static/trees"
@router.get("/predict", response_class=HTMLResponse)
async def view_home(request: Request):
    files = None
    if os.path.exists(TREE_DIR):
        print("exists")
        files = os.listdir(TREE_DIR)
    print(files)
    return templates.TemplateResponse("predict.html", {"request": request, "files":files})