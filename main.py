from fastapi import FastAPI
from nlp.sentiment_analysis import sentiment_score

analysis_api = FastAPI()

######### run fastapi ########
# uvicorn main:analysis_api --reload

@analysis_api.get("/sentiment/{sentence}")
async def sentiment_analysis(sentence):
    return sentiment_score(sentence,0.3,-0.3)
