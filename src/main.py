import uvicorn
import base64
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

from fastapi import FastAPI, UploadFile,File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from process import Mitosisdetection

# helper function for GPU config for Keras
def gpu_setup():
    gpus = tf.config.list_physical_devices(device_type = 'GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

# instantiate the app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# call GPU setup helper
gpu_setup()

#HTML Template Directory
templates = Jinja2Templates(directory="templates")

# instantiate the Mitosisdetection class
md = Mitosisdetection()

# create a route
@app.get('/')
def read_form():
    return 'Lets begin !'

# publish default html template
@app.get("/images", response_class = HTMLResponse)
async def read_random_file(request: Request):
    result = ""
    return templates.TemplateResponse('myhtml.html', context={'request': request, 'result': result})

# create image to predictions route
@app.post("/images")
async def predict_image(request: Request, selectFile: UploadFile = File(...)):
    if not selectFile:
        return {'predClass': 'No file uploaded to process'}
    else:
        img_content = await selectFile.read()
        
        #process input image through pipeline
        # uploaded image to numpy array
        img = Image.open(selectFile.file).convert('RGB')
        img = np.array(img)

        # format input/uploaded image to be returned to UI
        input_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        res, input_image = cv2.imencode(".jpg", input_image)
        base64_encoded_img = base64.b64encode(input_image.tobytes()).decode("utf-8")

        # make predictions on image
        # result dictionary and superimposed image
        result, simg = md.predict(img)
        
        # format superimposed image
        res, simg = cv2.imencode(".jpg", simg)

        #get heatmap image
        base64_hm_img = base64.b64encode(simg.tobytes()).decode("utf-8")

        predictions = ["0","1","0"]
        explanations = ["text1","text2","text3"]
        xcoordinates = [{400},{10},{250}]
        ycoordinates = [{400},{10},{250}]
        return templates.TemplateResponse('myhtml.html', context={'request': request, 'input_img': base64_encoded_img, 'heatmap_img': base64_hm_img, 'predDict': predictions,'explainDict': explanations, 'xcoords': xcoordinates, 'ycoords': ycoordinates})

    # run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

#terminal command: uvicorn src.main:app --reload --workers 1 --host 0.0.0.0 --port 8000