pip install opencv-python-headless

import cv2
import dlib
import numpy as np
from PIL import Image
import streamlit as st

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_landmarks(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None

    shape = predictor(gray, rects[0])
    landmarks = [(i, shape.part(i).x, shape.part(i).y) for i in range(68)]
    return landmarks


def visualize_landmarks(image, landmarks, index1, index2, index3, index4):
    image_copy = np.array(image).copy()

    # Draw landmarks
    for (i, (index, x, y)) in enumerate(landmarks):
        cv2.circle(image_copy, (x, y), 2, (0, 255, 0), -1)  # Green color for landmarks
        cv2.putText(image_copy, str(index), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw lines for distances
    point1_x, point1_y = landmarks[index1][1], landmarks[index1][2]
    point2_x, point2_y = landmarks[index2][1], landmarks[index2][2]
    point3_x, point3_y = landmarks[index3][1], landmarks[index3][2]
    point4_x, point4_y = landmarks[index4][1], landmarks[index4][2]

    # Line color and thickness
    line_color = (0, 0, 255)  # Red color
    line_thickness = 2

    # Draw lines
    cv2.line(image_copy, (point1_x, point1_y), (point2_x, point2_y), line_color, line_thickness)
    cv2.line(image_copy, (point3_x, point3_y), (point4_x, point4_y), line_color, line_thickness)

    return Image.fromarray(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))


def main():
    st.title("Facial Landmark Detection and Distance Comparison")
    st.write(
        "Upload an image to detect and visualize facial landmarks. Select four landmarks to compare their distances and see the difference.")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")

        st.sidebar.header("Landmark Selection")
        index1 = st.sidebar.slider("Select Landmark Index 1", min_value=0, max_value=67, value=0)
        index2 = st.sidebar.slider("Select Landmark Index 2", min_value=0, max_value=67, value=1)
        index3 = st.sidebar.slider("Select Landmark Index 3", min_value=0, max_value=67, value=2)
        index4 = st.sidebar.slider("Select Landmark Index 4", min_value=0, max_value=67, value=3)

        landmarks = get_landmarks(image)

        if landmarks is None:
            st.error("Could not detect landmarks in the image.")
        else:
            point1 = (landmarks[index1][1], landmarks[index1][2])
            point2 = (landmarks[index2][1], landmarks[index2][2])
            point3 = (landmarks[index3][1], landmarks[index3][2])
            point4 = (landmarks[index4][1], landmarks[index4][2])

            distance1 = np.linalg.norm(np.array(point1) - np.array(point2))
            distance2 = np.linalg.norm(np.array(point3) - np.array(point4))
            distance_diff = abs(distance1 - distance2)

            st.image(visualize_landmarks(image, landmarks, index1, index2, index3, index4),
                     caption="Image with Landmarks and Distances", use_column_width=True)
            st.write(f"Distance between Landmark {index1} and Landmark {index2}: {distance1:.2f} pixels")
            st.write(f"Distance between Landmark {index3} and Landmark {index4}: {distance2:.2f} pixels")
            st.write(f"Difference between the two distances: {distance_diff:.2f} pixels")


if __name__ == "__main__":
    main()
