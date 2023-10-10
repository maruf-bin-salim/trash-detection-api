from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from fastai.vision.all import *
from urllib.request import urlopen
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
async def get_net_image_prediction(request: Request):
    body = await request.json()
    image_link = body.get("image_link")
    print("Image link is -> ", image_link)

    if(image_link == None):
        return {"message": "No image link provided"}
    if image_link == "":
        return {"message": "No image link provided"}


    isValidUrl = False
    try:
        urlopen(image_link)
        isValidUrl = True
    except:
        isValidUrl = False

    if not isValidUrl:
        return {"message": "Invalid image link provided"}

    pred, idx, prob = learn.predict(PILImage.create(urlopen(image_link)))

    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    overall_probabilities = [{"class": classes[i], "probability": float(prob[i])} for i in range(len(classes))]
    overall_probabilities = sorted(overall_probabilities, key=lambda k: k['probability'], reverse=True)

    return {"prediction" : {"name": pred, "probability": float(prob[idx])}, "overall_probabilities": overall_probabilities}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    run(app, host="0.0.0.0", port=port)
