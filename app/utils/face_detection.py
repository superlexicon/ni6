import cv2
import numpy as np
from typing import List, Dict, Any


class FaceDetector:
    """Simple face detection using OpenCV's Haar cascades"""

    def __init__(self):
        # Load the Haar cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image using OpenCV Haar cascades.

        Args:
            image: Input image in BGR format (numpy array)

        Returns:
            List of face detection results with bounding boxes, confidence, and cropped faces
        """
        try:
            # Convert to grayscale for detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            face_results = []
            for (x, y, w, h) in faces:
                # Basic confidence based on face size
                confidence = min(w * h / (image.shape[0] * image.shape[1]) * 10, 1.0)

                # Crop the face region
                cropped_face = image[y:y+h, x:x+w]

                face_result = {
                    'bbox': [int(x), int(y), int(x+w), int(y+h)],  # [x1, y1, x2, y2]
                    'confidence': float(confidence),
                    'cropped_face': cropped_face,
                    'image_size': [image.shape[1], image.shape[0]],  # [width, height]
                    'extraction_method': 'opencv_haarcascade'
                }
                face_results.append(face_result)

            return face_results

        except Exception as e:
            print(f"Face detection failed: {str(e)}")
            return []