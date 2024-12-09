# app.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import io,os
import sys,json
from PIL import Image
import pandas as pd
import tensorflow as tf
import joblib
from tensorflow.keras.layers import Layer, Dense, Input, Dropout, LayerNormalization, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import GlobalAveragePooling1D 
from tensorflow.keras.layers import Embedding

from servermodel.classes import *
import requests
 # or GlobalAveragePooling2D

# Load the pre-trained TensorFlow model
loaded_model = tf.keras.models.load_model(
    './servermodel/best_vit_modelxxx.keras',
    custom_objects={
        'gelu': gelu,
        'PatchExtractor': PatchExtractor,
        'PatchEmbedding': PatchEmbedding,
        'PositionalEmbedding': PositionalEmbedding,
        'TransformerEncoder': TransformerEncoder,
        'GlobalAveragePoolingLayer': GlobalAveragePoolingLayer,
        'focal_loss_fixed': focal_loss(gamma=2., alpha=0.25)
    }
)

recommend=joblib.load("./servermodel/random_forest_model_recomendation.pkl")
fertilizer=joblib.load("./servermodel/random_forest_model_ferlilizer.pkl")

ModelDict={
    'rice':load_model('./servermodel/crop_disease_model_rice_vgg16.keras'),
    "apple":load_model('./servermodel/crop_disease_model_apple_vgg16.keras'),
    'pepper':load_model("./servermodel/best_model_pepper_vgg16.keras"),
    'corn':load_model("./servermodel/crop_disease_model_corn_vgg16.keras"),
    'tomato':load_model("./servermodel/best_vit_model_tomato_sunday.keras",
                        custom_objects={
                                            'gelu': gelu,
                                            'PatchExtractor': PatchExtractor,
                                            'PatchEmbedding': PatchEmbedding,
                                            'PositionalEmbedding': PositionalEmbedding,
                                            'TransformerEncoder': TransformerEncoder,
                                            'GlobalAveragePoolingLayer': GlobalAveragePoolingLayer,
                                            'focal_loss_fixed': focal_loss(gamma=2., alpha=0.25)
                                        }),
    'wheat':load_model("./servermodel/crop_disease_model_wheat_vgg16_ivattu.keras"),
    "potato":load_model("./servermodel/crop_disease_model_potato_vgg16_sunday.keras")
}

json_class={}
with open("./classes.json",'r')as j:
    json_class=json.load(j)

resposnse=requests.post("http://localhost:8080/local/registerType",json={"types":list(ModelDict.keys())})
if not resposnse.ok:
    sys.exit(200)

# Initialize the FastAPI app
app = FastAPI()

class CropRecommendationRequest(BaseModel):
    n: float  # Nitrogen content
    p: float  # Phosphorus content
    k: float  # Potassium content
    temperature: float  # Temperature in Celsius
    humidity: float  # Relative Humidity in percentage
    ph: float  # pH value of the soil
    rainfall: float  # Rainfall in mm

class FertilizerReommendation(BaseModel):
    n: float  
    p: float  
    k: float

# Preprocess function to prepare the image for prediction
def preprocess_image_cnn(image_data: bytes):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))  # Resize to model's input size
    img_array = img_to_array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def preprocess_image_vit(image_data: bytes, target_size=(224, 224)):
    # Load the image from bytes
    img = Image.open(io.BytesIO(image_data))
    
    # Resize the image to the target size
    img = img.resize(target_size)
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Scale the image to [-1, 1]
    img_array = (img_array / 127.5) - 1
    
    # Expand dimensions to make it batch of 1
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Endpoint to receive an image and optional description for prediction
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),  # Required image file
    c_type: str = Form(None)  # Optional string description
):
    try:
        print(c_type)
        result = {"max": None, "confidence":None, "id":""}

        # Read the image data
        image_data = await image.read()

        if c_type:
            if c_type not in ModelDict:
                raise HTTPException(status_code=400, detail=f"Model type '{c_type}' not found.")
            
            # Get predictions from the selected model
            predictions = ModelDict[c_type].predict(preprocess_image_cnn(image_data=image_data))
        else:
            # Get predictions from the Vision Transformer model
            predictions = loaded_model.predict(preprocess_image_vit(image_data=image_data))

        # Get the class with the maximum probability
        result['max'] = np.argmax(predictions, axis=-1).item()  # Get the index of the highest probability
        result['confidence'] = np.array(predictions).flatten().tolist()[result['max']] * 100
        result['id'] = json_class[c_type if c_type else "all"][result['max']]
      
        print(result)
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
    
@app.post("/predict/crop")
async def crop(req :CropRecommendationRequest):
    new_data_df = pd.DataFrame([[
        req.n,
        req.p,
        req.k,
        req.temperature,
        req.humidity,
        req.ph,
        req.rainfall
    ]], 
    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    try:
        prediction = recommend.predict(new_data_df)
        return {
            "crop": prediction[0]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")



@app.post("/predict/fertilizer")
async def crop(req :FertilizerReommendation):
    new_data_df = pd.DataFrame([[
        req.n,
        req.p,
        req.k,
    ]], 
    columns=["Nitrogen","Potassium","Phosphorous"])
    try:
        prediction = fertilizer.predict(new_data_df)
        return {
            "fertilizer": prediction[0]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")






