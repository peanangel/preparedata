from fastapi import FastAPI, Request
import cv2
import numpy as np
app = FastAPI()
import base64



def readb64(encoded_data):
    data = encoded_data.split(',',1)
    img_str = data[1]
    decode = base64.b64decode(img_str)
    img = cv2.imdecode(np.frombuffer(decode, np.uint8),cv2.COLOR_RGB2HSV)
    return img

def genImage(img):
    # Check if the input image is already grayscale (single-channel)
    if len(img.shape) == 2:
        # It's a grayscale image; no need to convert
        image = img
    else:
        # It's a color image; convert to grayscale
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = image.reshape(224,224,1)
    image = image.tolist()
    return image



@app.get("/")
def read_root():
    return {"result": "Hello, FastAPI!"}


@app.get("/api/genimage")
async def read_str(data: Request):
    try:
        json = await data.json()
        item_str = json["img"]
        img = readb64(item_str)
        image = genImage(img)
        # image = np.array(image,dtype='float32')
        # print(image.shape)
        return {"result": image}
    except ValueError as ve:
        print(f"Caught a ValueError: {ve}")
    except TypeError as te:
        print(f"Caught a TypeError: {te}")

    



