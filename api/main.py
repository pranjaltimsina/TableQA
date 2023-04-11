from io import StringIO
import pandas as pd
import torch
import requests
from transformers import TapasTokenizer, TapasForQuestionAnswering

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# set the allowed origins for CORS
origins = [
    "http://localhost",
    "http://localhost:5500",
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


output_model_file = '../models/tapas.bin'
model_name = "google/tapas-base-finetuned-sqa"
model = torch.load(output_model_file)
tokenizer = TapasTokenizer.from_pretrained(model_name)


@app.get("/")
def home():
  return {"Hello": "World"}

@app.post("/predict")
def predict(question: str, table: str):

  #  read the csv from the table where new lines are sperated by ; and values are seperated by ,
  table = table.replace(";", "\n")

  table = pd.read_csv(StringIO(table), sep=",")
  table = table.astype(str)
  question = [str(question)]
  inputs = tokenizer(table=table, queries=question, padding="max_length", return_tensors="pt")
  outputs = model(**inputs)

  ans = tokenizer.convert_logits_to_predictions(
      inputs, outputs.logits.detach()
  )
  ans = ans[0][0]
  answers = []
  print(ans)
  for x in ans:
      answers.append(str(table.iloc[x]))

  return {"answer": answers}

if __name__ == "__main__":
  uvicorn.run("main:app")
