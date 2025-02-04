from fastapi import FastAPI, UploadFile, File
import uvicorn
import tempfile
from pathlib import Path

import torchvision.io as tv_io
import torchvision.transforms.v2 as transforms
import torch

# Import the model class
from work.model import BuildingClassifier

# Constants
PATH = "model/ExportedModel.pth"
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# Load the model
LoadedModel = BuildingClassifier()  # Instantiate the model
LoadedModel.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))  # Load weights
LoadedModel.eval()  # Set the model to evaluation mode

labels = ['Bungalow', 'Highrise', 'Storey-Building']

app = FastAPI()

@app.get("/project")
def read_root():
    return {"Project": "Building Classification Project."}

# Preprocessing transform
pre_trans = transforms.Compose([
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
    transforms.CenterCrop((IMAGE_WIDTH, IMAGE_HEIGHT)),
])

def make_prediction(file_path):
    # Read and preprocess the image
    image = tv_io.read_image(file_path, tv_io.ImageReadMode.RGB)
    image = pre_trans(image).unsqueeze(0).to(torch.device('cpu'))  # Use 'cuda' if GPU is available

    # Make prediction
    with torch.no_grad():
        output = LoadedModel(image)
        prediction = torch.argmax(output, dim=1).item()

    return labels[prediction]

@app.post("/prediction")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        file_path = Path(temp_file.name)
        content = await file.read()
        temp_file.write(content)

    # Make prediction
    prediction = make_prediction(str(file_path))

    # Clean up the temporary file
    file_path.unlink()

    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")