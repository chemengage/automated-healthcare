import uvicorn
import base64
import numpy as np

from fastapi import FastAPI, UploadFile,File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


# from PIL import Image
#
# from src.process import Mitosisdetection
# import cv2
# import io
# import torch
# from torch import autocast


# instantiate the app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


#HTML Template Directory
templates = Jinja2Templates(directory="templates")

# instantiate the Mitosisdetection class
##md = Mitosisdetection()

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
        
        #return back input image
        base64_encoded_img = base64.b64encode(img_content).decode("utf-8")
        
        #process input image through pipeline
        # uploaded image to numpy array
        #img = Image.open(selectFile.file).convert('RGB')
        #heatmap = Image.open(selectFile.file).convert('RGB')
        #img = np.array(img)

        # make predictions on image
        # result dictionary
        # superimposed image = simg
        ## result, simg = md.predict(img)

        # format superimposed image
        ## simg = cv2.cvtColor(simg, cv2.COLOR_RGB2BGR)
        ## res, simg = cv2.imencode(".png", simg)
        #simg = simg.tobytes()   #convert to byte object
        
        #get heatmap image
        ##base64_hm_img = base64.b64encode(simg.tobytes()).decode("utf-8")

        predictions = ["0","1","0"]
        explanations = ["text1","text2","text3"]
        xcoordinates = [{400},{10},{250}]
        ycoordinates = [{400},{10},{250}]
        return templates.TemplateResponse('myhtml.html', context={'request': request, 'input_img': base64_encoded_img, 'heatmap_img': base64_encoded_img, 'predDict': predictions,'explainDict': explanations, 'xcoords': xcoordinates, 'ycoords': ycoordinates})

    # run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

#terminal command: uvicorn src.main:app --reload --workers 1 --host 0.0.0.0 --port 8000