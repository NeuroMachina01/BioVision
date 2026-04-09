from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# --- CORS CONFIGURATION ---
# --- CORS CONFIGURATION ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173", # Add this line!
    "http://127.0.0.1:5173", # Add this line just to be safe!
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. MODEL CONFIGURATION ---
# UPDATE THESE PATHS to your actual model files
PNEUMONIA_MODEL_PATH = r"C:\Users\puska\OneDrive\Desktop\BioVision\chest_xray\best_model.h5"
BRAIN_MODEL_PATH = r"C:\Users\puska\OneDrive\Desktop\BioVision\Brain MRI\brain_tumor_model.keras"
DR_MODEL_PATH = r"C:\Users\puska\OneDrive\Desktop\BioVision\Retina\diabetic_retinopathy_model.keras" # Use .keras if possible

# --- 2. LOAD MODELS ---
print("Loading models... (This may take a moment)")

# We wrap in try-except to prevent the whole app from crashing if one model is missing
try:
    MODEL_PNEUMONIA = tf.keras.models.load_model(PNEUMONIA_MODEL_PATH)
    print("✅ Pneumonia model loaded.")
except Exception as e:
    print(f"❌ Error loading Pneumonia model: {e}")
    MODEL_PNEUMONIA = None

try:
    MODEL_BRAIN = tf.keras.models.load_model(BRAIN_MODEL_PATH)
    print("✅ Brain Tumor model loaded.")
except Exception as e:
    print(f"❌ Error loading Brain Tumor model: {e}")
    MODEL_BRAIN = None

try:
    MODEL_DR = tf.keras.models.load_model(DR_MODEL_PATH)
    print("✅ DR model loaded.")
except Exception as e:
    print(f"❌ Error loading DR model: {e}")
    MODEL_DR = None


# --- 3. DEFINE CLASS NAMES ---
# Ensure these match the exact order of your training folders!

CLASS_NAMES_PNEUMONIA = ['NORMAL', 'PNEUMONIA']

CLASS_NAMES_BRAIN = [
    'Glioma Tumor', 
    'Meningioma Tumor', 
    'No Tumor', 
    'Pituitary Tumor'
] 

# Diabetic Retinopathy Classes (Full Names)
# Order: Mild, Moderate, No_DR, Proliferate_DR, Severe
CLASS_NAMES_DR = [
    'Mild Diabetic Retinopathy', 
    'Moderate Diabetic Retinopathy', 
    'No Diabetic Retinopathy', 
    'Proliferative Diabetic Retinopathy', 
    'Severe Diabetic Retinopathy'
]
CLASS_NAMES_BRAIN = [
    'Glioma Tumor', 
    'Meningioma Tumor', 
    'No Tumor', 
    'Pituitary Tumor'
] 

# Diabetic Retinopathy Classes (Full Names)
# Order: Mild, Moderate, No_DR, Proliferate_DR, Severe
CLASS_NAMES_DR = [
    'Mild Diabetic Retinopathy', 
    'Moderate Diabetic Retinopathy', 
    'No Diabetic Retinopathy', 
    'Proliferative Diabetic Retinopathy', 
    'Severe Diabetic Retinopathy'
]
CLASS_NAMES_BRAIN = [
    'Glioma Tumor', 
    'Meningioma Tumor', 
    'No Tumor', 
    'Pituitary Tumor'
] 

# Diabetic Retinopathy Classes (Full Names)
# Order: Mild, Moderate, No_DR, Proliferate_DR, Severe
CLASS_NAMES_DR = [
    'Mild Diabetic Retinopathy', 
    'Moderate Diabetic Retinopathy', 
    'No Diabetic Retinopathy', 
    'Proliferative Diabetic Retinopathy', 
    'Severe Diabetic Retinopathy'
]
CLASS_NAMES_BRAIN = [
    'Glioma Tumor', 
    'Meningioma Tumor', 
    'No Tumor', 
    'Pituitary Tumor'
] 

# Diabetic Retinopathy Classes (Full Names)
# Order: Mild, Moderate, No_DR, Proliferate_DR, Severe
CLASS_NAMES_DR = [
    'Mild Diabetic Retinopathy', 
    'Moderate Diabetic Retinopathy', 
    'No Diabetic Retinopathy', 
    'Proliferative Diabetic Retinopathy', 
    'Severe Diabetic Retinopathy'
]


# --- 4. HELPER FUNCTIONS ---

def read_file_as_image(data, target_size) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        image = image.resize(target_size)
        img_array = np.array(image).astype("float32") / 255.0
        return img_array
    except Exception as e:
        print("Error reading image:", e)
        return None

def prediction_process(model, class_names, image_data):
    """
    Generic function to handle prediction for any loaded model.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded successfully on server.")

    # 1. Prepare Image
    # Get input shape from model (e.g., (None, 256, 256, 3))
    input_shape = model.input_shape
    height = input_shape[1]
    width = input_shape[2]
    
    image = read_file_as_image(image_data, target_size=(height, width))
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_batch = np.expand_dims(image, 0) # Shape: (1, H, W, 3)

    # 2. Predict
    preds = model.predict(img_batch)
    
    # 3. Decode Results
    # Case A: Binary (Sigmoid) -> Output shape (1, 1)
    if preds.ndim == 2 and preds.shape[1] == 1:
        prob = float(preds[0][0])
        # Assuming Class 1 is the "Positive" case
        index = 1 if prob >= 0.5 else 0
        conf = prob if index == 1 else 1.0 - prob
        
        # Create dictionary for all probabilities
        all_probs = {
            class_names[0]: 1.0 - prob,
            class_names[1]: prob
        }
    
    # Case B: Multi-class (Softmax) -> Output shape (1, N)
    else:
        probs_list = preds[0].tolist()
        index = int(np.argmax(probs_list))
        conf = float(probs_list[index])
        
        all_probs = {}
        for i, prob_val in enumerate(probs_list):
            name = class_names[i] if i < len(class_names) else f"Class {i}"
            all_probs[name] = prob_val

    predicted_class = class_names[index] if index < len(class_names) else f"Class {index}"

    return {
        "class": predicted_class,
        "confidence": conf,
        "all_probabilities": all_probs
    }


# --- 5. ENDPOINTS ---

@app.get("/ping")
async def ping():
    return {"message": "Server is running"}

@app.post("/predict/pneumonia")
async def predict_pneumonia(file: UploadFile = File(...)):
    data = await file.read()
    return prediction_process(MODEL_PNEUMONIA, CLASS_NAMES_PNEUMONIA, data)

@app.post("/predict/brain_tumor")
async def predict_brain(file: UploadFile = File(...)):
    data = await file.read()
    return prediction_process(MODEL_BRAIN, CLASS_NAMES_BRAIN, data)

@app.post("/predict/dr")
async def predict_dr(file: UploadFile = File(...)):
    data = await file.read()
    return prediction_process(MODEL_DR, CLASS_NAMES_DR, data)


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)