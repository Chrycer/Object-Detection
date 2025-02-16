import cv2
import json
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify

# Initialize Firebase with Firestore
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load the YOLO model
model = YOLO('yolo11m.pt')

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image.")
        return

    # Perform object detection
    results = model.predict(source=image, save=False, save_txt=False, conf=0.5)

    # Extract detected objects and their confidence scores
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Class ID
            confidence = float(box.conf[0])  # Confidence score
            class_name = model.names[class_id]  # Get class name from YOLO model

            detection = {
                "object": class_name,
                "confidence": confidence
            }
            detections.append(detection)

    # Save results to Firestore
    doc_ref = db.collection('detections').document(image_path)
    doc_ref.set({"results": detections})
    print(f"Results saved to Firestore for image: {image_path}")

# Flask App to create API
app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    image_path = request.json.get('image_path')
    if not image_path:
        return jsonify({"error": "No image path provided"}), 400

    process_image(image_path)
    return jsonify({"status": "Detection completed"}), 200

if __name__ == '__main__':
    app.run(debug=True)
