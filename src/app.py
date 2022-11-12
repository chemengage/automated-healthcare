from fastapi import FastAPI, UploadFile, File
from starlette.responses import StreamingResponse
from PIL import Image
import numpy as np
from torch import autocast
import uvicorn
import cv2
import io
import torch
from process import Mitosisdetection

# instantiate the app
app = FastAPI()

# instantiate the Mitosisdetection class
md = Mitosisdetection()

# create a route
@app.get("/")
def index():
    return {"text" : "We're running!"}

# create img route
@app.post("/image")
def img_upload(img: UploadFile = File(...)):
    # uploaded image to numpy array
    img = Image.open(img.file).convert('RGB')
    img = np.array(img)

    # make predictions on image
    # result dictionary
    # superimposed image = simg
    result, simg = md.predict(img)

    # format superimposed image
    simg = cv2.cvtColor(simg, cv2.COLOR_RGB2BGR)
    res, simg = cv2.imencode(".png", simg)

    return StreamingResponse(io.BytesIO(simg.tobytes()), media_type="image/png")

# run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)