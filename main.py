import string
from fastapi import FastAPI
from typing import List
from nlp.sentiment_analysis import sentiment_list_score
from pydantic import BaseModel

analysis_api = FastAPI()

class Data(BaseModel):
    sentences: List[str]

######### run fastapi ########
# uvicorn main:analysis_api --reload

@analysis_api.post("/sentiment")
async def sentiment_analysis(data: Data):
    ret = sentiment_list_score(data.sentences,0.3,-0.3)
    print(ret)
    return ret


# class GraphList(BaseModel):
#     data: List[GraphBase]

# @app.post("/dummypath")
# async def get_body(data: GraphList):
#     return data