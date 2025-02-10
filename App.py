import streamlit as st
import os
import tempfile
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import pickle
import numpy as np
from torch import nn

model_path =  "cnn_model/cnnmodel_model.pth"
metadata_path = "metadata.csv"
encoders_path = 'Ring_resources/encoders_ring.pkl'

# Set Streamlit wide mode without navigation bar
st.set_page_config(layout="wide", page_title="Jewellery Classification", page_icon="💎", initial_sidebar_state="collapsed")

# Load the model
class CNNModel(nn.Module):
    def __init__(self, num_classes_divison, num_classes_gender, num_classes_classification,
                 num_classes_make, num_classes_mastercollection,
                 num_classes_collection, num_classes_subclassification, num_classes_subsection,
                 num_classes_jewellery):
        super(CNNModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_divison = nn.Linear(self.resnet.fc.in_features, num_classes_divison)
        self.fc_gender = nn.Linear(self.resnet.fc.in_features, num_classes_gender)
        self.fc_classification = nn.Linear(self.resnet.fc.in_features, num_classes_classification)
        self.fc_make = nn.Linear(self.resnet.fc.in_features, num_classes_make)
        self.fc_mastercollection = nn.Linear(self.resnet.fc.in_features, num_classes_mastercollection)
        self.fc_collection = nn.Linear(self.resnet.fc.in_features, num_classes_collection)
        self.fc_subclassification = nn.Linear(self.resnet.fc.in_features, num_classes_subclassification)
        self.fc_subsection = nn.Linear(self.resnet.fc.in_features, num_classes_subsection)
        self.fc_jewellery = nn.Linear(self.resnet.fc.in_features, num_classes_jewellery)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        return {
            'devision': self.fc_divison(x),
            'gender': self.fc_gender(x),
            'classification': self.fc_classification(x),
            'make': self.fc_make(x),
            'mastercollection': self.fc_mastercollection(x),
            'collection': self.fc_collection(x),
            'subclassification': self.fc_subclassification(x),
            'subsection': self.fc_subsection(x),
            'jewellery': self.fc_jewellery(x)
        }

@st.cache_data
def load_encoders(filename=encoders_path):
    with open(filename, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_metadata():
    return pd.read_csv(metadata_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image_path, model, encoder):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        predictions = {label: torch.argmax(outputs[label], 1).item() for label in outputs}
    return {key: encoder[f'encoder_{key}'].inverse_transform([predictions[key]])[0] for key in predictions}

def main():
    st.title("💎 Jewellery Image Classification")
    st.write("Upload an image to classify it into jewellery categories.")
    
    model = CNNModel(1, 2, 1, 1, 2, 5, 3, 5, 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    encoder = load_encoders()
    metadata = load_metadata()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        image = Image.open(tmp_file_path).convert('RGB')
        image = image.resize((500, 350))  # Ensure consistent width and height
        
        file_name = uploaded_file.name
        design_number = os.path.splitext(file_name)[0]
        
        # Define the target columns for display
        target_columns = ['devision', 'gender', 'classification', 'make', 'mastercollection', 
                          'collection', 'subclassification', 'subsection', 'jewellery']

        actual_data_columns = ['Division', 'Gender', 'Classification', 'Make', 'MasterCollection',  
                               'Collection', 'SubClassification', 'SubSection', 'Jewellery']
        
        actual_labels = {}
        if design_number in metadata["DesignNumber"].values:
            row = metadata[metadata["DesignNumber"] == design_number].iloc[0]
            actual_labels = {col: row[actual_data_columns[i]] for i, col in enumerate(target_columns)}
        else:
            st.warning(f"Actual labels not found for design number: {design_number}")
        
        predictions = predict(tmp_file_path, model, encoder)
        os.unlink(tmp_file_path)
        
        col1, col2 = st.columns([2, 2])
        
        with col1:
            st.image(image, caption="Uploaded Image", width=500, use_container_width=False)
        
        with col2:
            st.subheader("📊 Predictions vs Actual Labels")
            df = pd.DataFrame({
                "Label": actual_data_columns,
                "Prediction": [predictions.get(col, "N/A") for col in target_columns],
                "Actual": [actual_labels.get(col, "N/A") for col in target_columns]
            })
            st.dataframe(data=df, height=350, width=500)

if __name__ == "__main__":
    main()
