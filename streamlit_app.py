import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import requests
import os
import av

from streamlit_webrtc import webrtc_streamer

# Create application title and file uploader widget.
st.title("Face Detection App")
img_file_buffer = st.file_uploader("Escolha uma imagem", type=['jpg', 'jpeg', 'png'])

def process_videoframe(frame):  
    # Convert the frame to an OpenCV image.
    img = frame.to_ndarray(format="bgr24")
    
    # Load the DNN model if not already loaded.
    if 'net' not in st.session_state:
        st.session_state.net = load_model()
    
    # Detect faces in the image.
    detections = detectFaceOpenCVDnn(st.session_state.net, img)
    
    # Process the detections and draw bounding boxes.
    img, _ = process_detections(img, detections, conf_threshold=0.5)
    
    # Convert the processed image back to a VideoFrame.
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="streamer",
    video_frame_callback=process_videoframe,
    sendback_audio=False,
    # media_stream_constraints={
    #     "video": {
    #         "width": 1280,
    #         "height": 720
    #     }
    # }
)

def detectFaceOpenCVDnn(net, frame):
    # Create a blob from the image and apply some pre-processing.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Set the blob as input to the model.
    net.setInput(blob)
    # Get Detections.
    detections = net.forward()
    return detections


# Function for annotating the image with bounding boxes for each detected face.
def process_detections(frame, detections, conf_threshold=0.5):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    # Loop over all detections and draw bounding boxes around each face.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])
            bb_line_thickness = max(1, int(round(frame_h / 200)))
            # Draw bounding boxes around detected faces.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bb_line_thickness, cv2.LINE_8)
    return frame, bboxes


# Function to load the DNN model.
@st.cache_resource()
def load_model(): 
    # Create the directory if it doesn't exist 
    directory = "models"
    os.makedirs(directory, exist_ok=True) 

    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    modelFile = os.path.join(directory, modelFile) 

    configFile = "deploy.prototxt.txt" 
    configFile = os.path.join(directory, configFile) 

    # Check if the model files exist, if not, download them.
    with st.spinner("Loading DNN model...", show_time=True):
    
        if not (os.path.exists(modelFile) and os.path.exists(configFile)):
            # Download the model files if they are not present.
            model_url = "https://github.com/sr6033/face-detection-with-OpenCV-and-DNN/raw/refs/heads/master/res10_300x300_ssd_iter_140000.caffemodel"
            config_url = "https://github.com/sr6033/face-detection-with-OpenCV-and-DNN/raw/refs/heads/master/deploy.prototxt.txt"
            
            st.info("Model not found. Downloading model files from GitHub...")
            if not (requests.get(model_url).ok and requests.get(config_url).ok):
                st.error("Failed to download model files.")
                return None    
            if not (os.path.exists(modelFile) and os.path.exists(configFile)):
                with open(modelFile, 'wb') as f:
                    f.write(requests.get(model_url).content)
                with open(configFile, 'wb') as f:
                    f.write(requests.get(config_url).content)

        # Load the model using OpenCV's DNN module.
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        st.success("Model loaded successfully!")
        return net


# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href


net = load_model()

if img_file_buffer is not None:
    # Read the file and convert it to opencv Image.
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    # Loads image in a BGR channel order.
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    # Or use PIL Image (which uses an RGB channel order)
    # image = np.array(Image.open(img_file_buffer))

    # Create placeholders to display input and output images.
    placeholders = st.columns(2)
    # Display Input image in the first placeholder.
    placeholders[0].image(image, channels='BGR')
    placeholders[0].text("Input Image")

    # Create a Slider and get the threshold from the slider.
    conf_threshold = st.slider("SET Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=0.5)

    # Call the face detection model to detect faces in the image.
    detections = detectFaceOpenCVDnn(net, image)

    # Process the detections based on the current confidence threshold.
    out_image, _ = process_detections(image, detections, conf_threshold=conf_threshold)

    # Display Detected faces.
    placeholders[1].image(out_image, channels='BGR')
    placeholders[1].text("Output Image")

    # Convert opencv image to PIL.
    out_image = Image.fromarray(out_image[:, :, ::-1])
    # Create a link for downloading the output file.
    st.markdown(get_image_download_link(out_image, "face_output.jpg", 'Download Output Image'), unsafe_allow_html=True)