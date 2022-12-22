import string
from fastapi import FastAPI, Request
from typing import List
from nlp.sentiment_analysis_en import sentiment_list_score_en
from nlp.sentiment_analysis_ko import sentiment_list_score_ko
from pydantic import BaseModel
import asyncio

import logging

from pydantic import BaseModel
from pydantic.class_validators import root_validator
from pythonjsonlogger import jsonlogger


formatter = jsonlogger.JsonFormatter()

logHandler = logging.StreamHandler()
logHandler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logHandler)

analysis_api = FastAPI()

class Data(BaseModel):
    sentences: List[str]

######### run fastapi ########
# uvicorn main:analysis_api --reload

@analysis_api.middleware("http")
async def log_stuff(request: Request, call_next):
    logger.debug(f"{request.method} {request.url}")
    # body = await request.body()
    # logger.debug(f"{body}")
    response = await call_next(request)
    logger.debug(response)
    logger.debug(response.status_code) 
    return response

@analysis_api.post("/sentiment/{language}")
async def sentiment_analysis(data: Data, language:str):
    print(data.sentences)
    if language=="ko":
        ret = sentiment_list_score_ko(data.sentences)
    else:
        ret = sentiment_list_score_en(data.sentences,0.3,-0.3)
    return ret

    
  

# class GraphList(BaseModel):
#     data: List[GraphBase]

# @app.post("/dummypath")
# async def get_body(data: GraphList):
#     return data