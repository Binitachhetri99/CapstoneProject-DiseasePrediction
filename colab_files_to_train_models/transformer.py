import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Assuming class_names is defined earlier in your script
# Example:
label_mapping = {
    'Melanoma': 1,
    'Basal cell carcinoma': 0,
    'Melanocytic nevi': 2,
    'Benign keratosis-like lesions': 3,
    'Actinic keratoses and intraepithelial carcinoma': 6,
    'Dermatofibroma': 5,
    'Vascular lesions': 4
}

# Reverse the label mapping to map index to label name
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Define the image transformation process
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    # Initialize the model with the correct architecture
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch32-384", 
        num_labels=7,  # Adjust to match the number of classes in your dataset
        ignore_mismatched_sizes=True  # Correct keyword for handling size mismatches
    )
    
    # Load the saved weights into the model
    state_dict = torch.load("/Users/binitachhetri/Downloads/CapstoneProject-DiseasePrediction/colab_files_to_train_models/model_weights3.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    model.eval()  # Set the model to evaluation mode
    return model

# Streamlit UI
st.title("Skin Lesion Classification")
st.write("Upload an image to classify the skin lesion.")

# Add a reset button to clear the current session state
if st.button("Reset"):
    st.session_state.clear()  # Clears session state to reset the app

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if an image was uploaded
if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Apply the transformation
    transformed_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Load the model
    model = load_model()

    # Make prediction
    with torch.no_grad():
        outputs = model(transformed_image).logits  # Forward pass
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence_score = probabilities[0, predicted_class].item()
        predicted_class_name = reverse_label_mapping[predicted_class]

    # Display the result
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"### Predicted Class: {predicted_class_name}")
    st.write(f"### Confidence Score: {confidence_score:.2f}")

    # Visualize the denormalized image
    transformed_image_display = transformed_image.squeeze(0).permute(1, 2, 0).numpy()
    transformed_image_display = (transformed_image_display * 0.5) + 0.5  # Denormalize

    fig, ax = plt.subplots()
    ax.imshow(transformed_image_display)
    ax.axis('off')
    ax.set_title(f"Predicted: {predicted_class_name} | Confidence: {confidence_score:.2f}")
    st.pyplot(fig)
