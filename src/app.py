
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from train import MultiModalClassifier

# Page configuration
st.set_page_config(page_title="Multi-Modal Defect Inspection", layout="wide")

st.title("Multi-Modal Surface Defect Inspection System")
st.markdown("Early Fusion AI for Automated Defect Detection using Computer Vision and Sensor Telemetry.")

# Load the pre-trained model
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = MultiModalClassifier(num_sensor_features=4).to(device)
    
    # Define the model path relative to the current file
    model_path = Path(__file__).resolve().parent.parent / "models" / "multimodal_model.pth"
    
    # Check if the model file exists
    if not model_path.exists():
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()  
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device
model, device = load_model()

# Sidebar for sensor telemetry inputs
st.sidebar.header("Sensor Telemetry Inputs")
temp = st.sidebar.slider("Temperature (°C)", 20.0, 120.0, 75.0)
vibration = st.sidebar.slider("Vibration (mm/s)", 0.0, 10.0, 2.5)
pressure = st.sidebar.slider("Pressure (PSI)", 50.0, 150.0, 100.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 1.0, 0.4)

# File uploader for surface images
uploaded_file = st.file_uploader("Upload Surface Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Processing images
    pil_img = Image.open(uploaded_file).convert('RGB')
    rgb_img = np.array(pil_img)
    rgb_img_resized = cv2.resize(rgb_img, (224, 224))
    rgb_img_float = np.float32(rgb_img_resized) / 255.0

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(rgb_img_resized).unsqueeze(0).to(device)

    # 1. Scaling Sensor Inputs 
    temp_scaled = (temp - 70.0) / 25.0
    vib_scaled = (vibration - 2.5) / 2.0
    press_scaled = (pressure - 100.0) / 20.0
    hum_scaled = (humidity - 0.5) / 0.2

    sensor_tensor = torch.tensor([[temp_scaled, vib_scaled, press_scaled, hum_scaled]], dtype=torch.float32).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor, sensor_tensor)
        
    probs = torch.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)

    confidence_score = confidence.item()
    label = predicted_class.item()

    # 2. Correct Class Evaluation
    is_defective = (label == 1)

    # Display results
    st.subheader("Inspection Verdict")
    col_res, col_conf = st.columns(2)
    
    if is_defective:
        col_res.error("DEFECT DETECTED")
    else:
        col_res.success("NO DEFECT DETECTED")
        
    col_conf.metric("Confidence Score", f"{confidence_score * 100:.2f}%")

    # Grad-CAM Explainability
    target_layers = [model.vision_backbone.features[-1]]

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, sensors):
            super().__init__()
            self.model = model
            self.sensors = sensors
        def forward(self, x):
            return self.model(x, self.sensors)

    wrapped_model = ModelWrapper(model, sensor_tensor)
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    # Display images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb_img_resized, caption="Uploaded Surface Image", use_container_width="stretch")
    with col2:
        st.image(visualization, caption="Grad-CAM Heatmap (Explainability)", use_container_width="stretch")

    # Additional Safety Check for Sensor Anomalies
    MAX_TEMP = 110.0
    MAX_VIB = 8.0

    # Warning for critical sensor readings is Abnormal regardless the image prediction
    if temp > MAX_TEMP or vibration > MAX_VIB:
        st.error("⚠️ DEFECT / SENSOR ANOMALY DETECTED (Critical Operating Conditions)")
    else:
        if is_defective:
            st.error("DEFECT DETECTED")
        else:
            st.success("NO DEFECT DETECTED")