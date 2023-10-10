from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from fastai.vision.all import *
from io import BytesIO
import os

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
async def get_net_image_prediction(file: UploadFile):
    image_bytes = await file.read()

    if not image_bytes:
        return {"message": "No image data provided"}

    try:
        image = PILImage.create(BytesIO(image_bytes))
    except Exception as e:
        return {"message": f"Error processing image: {str(e)}"}

    pred, idx, prob = learn.predict(image)

    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    overall_probabilities = [{"class": classes[i], "probability": float(prob[i])} for i in range(len(classes))]
    overall_probabilities = sorted(overall_probabilities, key=lambda k: k['probability'], reverse=True)

    return {"prediction" : {"name": pred, "probability": float(prob[idx])}, "overall_probabilities": overall_probabilities}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    run(app, host="0.0.0.0", port=port)
