import streamlit as st
import torch
from PIL import Image
from src.model import PlantDiseaseModel
from torchvision import transforms
from src.dataset import get_classes


## 1. Page Setup
st.set_page_config(page_title = "AgroVision-Dx", layout = "centered")
st.title('🌿 LeafDoctor: ViT-Powered Plant Disease Diagnosis')
st.write("Upload a photo of a plant leaf for disease diagnosis")

## 2. Load Model
@st.cache_resource # Stops reloading the model on every interaction
def load_trained_model(num_classes):

    model = PlantDiseaseModel(num_classes=num_classes)  # Adjust num_classes as per your dataset
    model.load_state_dict(torch.load('models/best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

class_names = get_classes()
num_classes = len(class_names)

model = load_trained_model(num_classes)

## 3. Image Upload
file = st.file_uploader("Select Leaf Image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, caption = "Uploaded Image", use_container_width=True)


    ## Preprocess Image
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    input_tensor = preprocess(img).unsqueeze(0)

    ## Prediction
    with torch.no_grad():
        output =  model(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        top_prob, top_idx = torch.max(prob, dim=1)

    confidence = top_prob.item() * 100
    result = class_names[top_idx.item()]

    st.subheader(f"Prediction: {result}")
    st.progress(confidence / 100)
    st.write(f"Confidence: {confidence:.2f}%")