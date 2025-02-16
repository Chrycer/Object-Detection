import cv2
import json
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore, storage
from flask import Flask, request, jsonify
from random import choices
from string import digits, ascii_letters
import os

# Initialize Firebase with Firestore and Storage
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'accenture-a5bf3.firebasestorage.app'
})
db = firestore.client()
bucket = storage.bucket()

# Load the YOLO model
model = YOLO('yolo11s.pt')

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image.")
        return

    # Perform object detection
    results = model.predict(source=image, save=False, save_txt=False, conf=0.5)

    # Extract detected objects and their coordinates
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Class ID
            class_name = model.names[class_id]  # Get class name from YOLO model
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            # Draw bounding box and label (without confidence)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detection = {
                "object": class_name,
                "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            }
            detections.append(detection)

    try:
        # Generate a unique name for the image and document
        name = ''.join(choices(ascii_letters + digits, k=8))
        annotated_image_path = f"{name}.jpg"
        
        # Save the annotated image locally
        cv2.imwrite(annotated_image_path, image)

        # Upload annotated image to Firebase Storage
        blob = bucket.blob(f"annotated_images/{name}.jpg")
        blob.upload_from_filename(annotated_image_path)
        blob.make_public()
        image_url = blob.public_url

        # Save detection results to Firestore
        doc_ref = db.collection('detections').document(name)
        doc_ref.set({
            "results": detections,
            "image_url": image_url
        })
        print(f"Results saved to Firestore and image uploaded: {image_url}")

        # Optionally, remove the local copy after uploading
        os.remove(annotated_image_path)

    except Exception as e:
        print(f"Error in Firebase operation: {e}")

# Flask App to create API
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to YOLO Detection API"}), 200

@app.route('/detect', methods=['POST'])
def detect():
    image_path = request.json.get('image_path')
    if not image_path:
        return jsonify({"error": "No image path provided"}), 400

    process_image(image_path)
    return jsonify({"status": "Detection completed"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
