import os, uuid
import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO('yolov8s.pt')
st.title('Object Detection App')
st.write('Upload image to detect object')

uploaded_file = st.file_uploader('Choose a picture', type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption = "Uploaded Photo", use_container_width=True)

    image_path = os.path.join("Result/img_" + f"{uuid.uuid4()}.jpg")
    image.save(image_path)

    results = model(image_path)
    results[0].save(filename=image_path)

    st.image(image_path, caption='Result after detection', use_container_width=True)

