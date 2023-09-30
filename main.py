from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from fastai.vision.all import *
import os
import base64
from PIL import Image
from io import BytesIO
import json



app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers,
)

learn = load_learner("export.pkl")

@app.get("/")
async def root():
    return {"message": "Welcome to the Garbage Classification API!"}

@app.post("/predict")
async def predict_from_data_url(request: Request):
    try:
        data = await request.json()
        image_data_url = data.get("image_data_url")

        if not image_data_url:
            raise HTTPException(status_code=422, detail="No image data URL provided")

        # Extract the base64-encoded image data from the Data URL
        if image_data_url.startswith("data:image/jpeg;base64,"):
            image_data = image_data_url.split(",")[1]
            image_bytes = base64.b64decode(image_data)

            if image_bytes is None or len(image_bytes) == 0:
                raise HTTPException(status_code=422, detail="Couldn't get image bytes from the data URL")
            
            bytesToImage = BytesIO(image_bytes)
            
            if bytesToImage is None:
                raise HTTPException(status_code=422, detail="Couldn't convert to bytesIO from the data URL")

            # Convert the image bytes to a PIL Image
            image = Image.open(bytesToImage)

            if image is None:
                raise HTTPException(status_code=422, detail="Couldn't convert to image!")


            # Perform prediction on the image using your model
            pred, idx, prob = learn.predict(image)

            classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
            overall_probabilities = [{"class": classes[i], "probability": float(prob[i])} for i in range(len(classes))]
            overall_probabilities = sorted(overall_probabilities, key=lambda k: k['probability'], reverse=True)

            return {"prediction": {"name": pred, "probability": float(prob[idx])}, "overall_probabilities": overall_probabilities}

        else:
            raise HTTPException(status_code=422, detail="Invalid image data URL format")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    run(app, host="0.0.0.0", port=port)
