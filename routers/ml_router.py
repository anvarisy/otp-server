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

new_request = {
        "pic_id": "085811751000",
        "purpose": "LOGIN",
        "latitude": -6.6502314,
        "longitude": 106.7560309,
        "device_name": "SM-A235F",
        "os_version": "14",
        "manufacturer": "samsung",
        "cpu_info": "Qualcomm Technologies, Inc KHAJE",
        "platform": "ANDROID",
        "ip": "192.168.1.34"
    }
fraud_request = {
        "pic_id": "085811751000",
        "purpose": "LOGIN",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "device_name": "Xiaomi Redmi Note 10",
        "os_version": "14",
        "manufacturer": "Xiaomi",
        "cpu_info": "Mediatek Helio G95",
        "platform": "ANDROID",
        "ip": "203.123.123.123"
    }
outlier_request = {
        "pic_id": "085811751000",
        "purpose": "LOGIN",
        "latitude": 30.6139,
        "longitude": 106.2090,
        "device_name": "Sony Experia",
        "os_version": "14",
        "manufacturer": "experia",
        "cpu_info": "Mediatek Helio G95",
        "platform": "ANDROID",
        "ip": "192.168.5.34"
    }


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
        rf.train(df)
        return {
            "error": "",
            "status": True,
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": False,
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