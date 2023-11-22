from fastapi import FastAPI, File, UploadFile
import requests
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI()

# URL de Google Drive para el archivo compartido
file_id = "1wMb03-UkWY2PmWkvZKUxXZppuINfOFza"
url = f"https://drive.google.com/uc?id={file_id}"

# Descargar el modelo
response = requests.get(url)
with open("modelo_temporal.h5", "wb") as file:
    file.write(response.content)

model = load_model("modelo_temporal.h5")

@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    # Lee y procesa la imagen
    image = Image.open(BytesIO(await file.read()))
    image = image.resize((100, 100)) # Asegúrate de cambiar el tamaño según tu modelo
    image_array = np.array(image)

    # Realiza la predicción
    prediction = model.predict(np.array([image_array]))
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Diccionario de vegetales con sus respectivos índices
    vegetables = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 
                  'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 
                  'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

    # Obtiene el nombre del vegetal predicho
    predicted_vegetable = vegetables[predicted_class_index]

    return {"predicted_vegetable": predicted_vegetable, "prediction": prediction.tolist()}