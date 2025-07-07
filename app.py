import cv2
import streamlit as st
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)
def detect_faces(color,scale_factor,min_neighbors):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    bgr_color = hex_to_bgr(color)
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h),bgr_color, 2)
            cv2.imwrite('face.jpg', frame[y:y+h, x:x+w])
        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")
    st.markdown("""
    ### Instructions
    1. Choose the color of the rectangle for face detection.
    2. Adjust the **scaleFactor** and **minNeighbors** parameters to improve detection.
    3. Click **Detect Faces** to process the image.
        """)
    color = st.color_picker("Pick A Color", "#00f900")
    scale_factor = st.slider("scaleFactor", min_value=1.05, max_value=2.0, value=1.1, step=0.05)
    min_neighbors = st.slider("minNeighbors", min_value=1, max_value=10, value=5, step=1)
    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        # Call the detect_faces function
        detect_faces(color,scale_factor,min_neighbors)
if __name__ == "__main__":
    app()