from fastapi import FastAPI
import uvicorn
from fastapi import UploadFile, File
import re

app = FastAPI()

@app.get("/project")
def read_root():
    return {"Project": "building Project."}


@app.post("/prediction")
async def make_prediction(file: UploadFile =  File(...)):
    # Read the file uploaded by user

    # Make Prediction
    path= re.sub(r'[\]+', '/', file)
    predictions= make_prediction(path)
    return(predictions)
    
if __name__ == "__main__":
    uvicorn.run(app, port=8080, host= "0.0.0.0")