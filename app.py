import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Load Haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Convert hex color to BGR
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


# Define VideoTransformer
class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self, color, scale_factor, min_neighbors):
        self.bgr_color = hex_to_bgr(color)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), self.bgr_color, 2)
            # Optionally save detected face:
            face_crop = img[y:y+h, x:x+w]
            cv2.imwrite('face.jpg', face_crop)

        return img


# Streamlit app
def app():
    st.title("Face Detection using Viola-Jones Algorithm with WebRTC")
    st.write("Face detection directly in your browser using your webcam!")

    st.markdown("""
    ### Instructions
    1. Pick the color of the rectangle for face detection.
    2. Adjust **scaleFactor** and **minNeighbors** for detection sensitivity.
    3. Start your webcam and see results in real-time!
    """)

    color = st.color_picker("Pick A Color", "#00f900")
    scale_factor = st.slider("scaleFactor", min_value=1.05, max_value=2.0, value=1.1, step=0.05)
    min_neighbors = st.slider("minNeighbors", min_value=1, max_value=10, value=5, step=1)

    st.write("Click **Start** below to activate your webcam:")

    webrtc_streamer(
        key="face-detection",
        video_transformer_factory=lambda: FaceDetectionTransformer(color, scale_factor, min_neighbors),
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True
    )


if __name__ == "__main__":
    app()
