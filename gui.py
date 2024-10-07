import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# Load the trained model
model = load_model('brain_tumor_vgg19_model.h5')

# Define class names
class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']

def preprocess_image(image):
    # Resize the image to 224x224 (VGG19 input size)
    image = image.resize((224, 224))
    # Convert to array and expand dimensions
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    # Normalize pixel values
    image_array = image_array / 255.0
    return image_array

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_index = np.argmax(prediction[0])
    confidence = prediction[0][class_index]
    return class_names[class_index], confidence

# Streamlit app
page = st.sidebar.radio("Hop in!", ['Tumor Detection', 'Performance metrics', 'Team'])

if page == 'Tumor Detection':
    st.title('Brain Tumor Detection with CNN')

    uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)
        
        if st.button('Predict'):
            class_name, confidence = predict(image)
            st.write(f"Prediction: {class_name}")
            st.write(f"Confidence: {confidence:.2%}")

    st.write("I can predict the tumor type in the brain MRI image ( Trained with VGG19 with about 8000 images )")

elif page == 'Performance metrics':
    st.title('Model Performance Visualizations')

    # List of visualization files
    viz_files = [
        'accuracy_loss_plot.png',
        'accuracy_vs_loss_plot.png',
        'confusion_matrix.png',
        'learning_rate_plot.png',
        'precision_recall_curve.png',
        'roc_curve.png',
        'training_progress_plot.png'
    ]

    # Display each visualization
    for viz_file in viz_files:
        if os.path.exists(f'visualizations/{viz_file}'):
            st.subheader(viz_file.replace('_', ' ').replace('.png', '').title())
            st.image(f'visualizations/{viz_file}', use_column_width=True)
        else:
            st.warning(f"File not found: {viz_file}")

    st.write("These visualizations provide insights into the model's training process and performance.")

elif page == 'Team':

    st.title('Team')
    st.subheader("Guide : Dr.R.Gnanakumari")
    st.subheader("Members:")
    st.subheader("Dharshan - 727821TUCS040")
    st.subheader("Gautham - 727821TUCS049")
    st.subheader("Jashwanth - 727821TUCS061")

    