from typing import Dict, List, Union
from fastapi.openapi.utils import get_openapi
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
import pandas as pd
from ml.isolation_forest import IsolationImplementaion
from ml.preprocess import Preprocess
from ml.random_forest import RandomForestImplementation
from sender.dto import LearnRequest, LearnResponse, RandomForestResponse, UserActionEntity
from sender.helper import generate_augmented_data, generate_seed_data
router = APIRouter()

# @router.get("/otp/learn/isolation")
# async def learn_isolation():
#     isf = IsolationImplementaion()
#     filename = "data.csv"
#     df = pd.read_csv(filename)
#     isf.train(df)

# @router.get("/otp/predict/isolation")
# async def predict_isolation():
#     isf = IsolationImplementaion()
#     normal = isf.predict(new_request)
#     fraud = isf.predict(fraud_request)
#     outlier = isf.predict(outlier_request)
#     return {"result":normal,
#             "fraud":fraud,
#             "outlier":outlier}

@router.post("/learn-random-forest", response_model=LearnResponse)
async def learn_random_forest(body: LearnRequest):
    rf = RandomForestImplementation()
    filename = f'./uploaded_files/{body.file_name}'
    df = pd.read_csv(filename)
    try:
        result = rf.train(df, body.tree_estimator, body.test_count)
        # Sertakan hasil yang dikembalikan oleh fungsi train dalam response
        return {
            "error": "",
            "status": True,
            "metrics": result['metrics'],
            "plot_filename": result['plot_filename'],
            "voting_results": result['voting_results'],
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": False,
            "metrics": {},
            "plot_filename": '',
            "voting_results": [],
        }
        
@router.post("/predict-random-forest", response_model=RandomForestResponse)
async def predict_random_forest(entity: UserActionEntity):
    rf = RandomForestImplementation()
    try:
        result = rf.predict(entity)
        return {
            "error": "",
            "status": True,
            "data": {
                "result": result
            }
        }
    except Exception as e:
        # Jika Anda ingin menampilkan error ke user, Anda bisa memasukkan 'str(e)' ke dalam "error"
        # Namun, berhati-hatilah karena terkadang pesan error bisa mengandung informasi sensitif
        return {
            "error": str(e),
            "status": False,
            "data": {
                "result": ""
            }
        }