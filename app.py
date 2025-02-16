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
        return None, None, None

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

        # Save detections as a JSON file locally
        json_filename = f"{name}.json"
        with open(json_filename, 'w') as json_file:
            json.dump({"detected_objects": detections, "image_url": image_url}, json_file)

        # Upload JSON file to Firebase Storage
        json_blob = bucket.blob(f"json_results/{json_filename}")
        json_blob.upload_from_filename(json_filename)
        json_blob.make_public()
        json_url = json_blob.public_url

        # Optionally, remove the local copies after uploading
        os.remove(annotated_image_path)
        os.remove(json_filename)

        # Return detections, image URL, and JSON URL
        return detections, image_url, json_url

    except Exception as e:
        print(f"Error in Firebase operation: {e}")
        return None, None, None

# Flask App to create API
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    # Retrieve the latest detection from Firestore
    try:
        # Get the most recent document from the 'detections' collection
        docs = db.collection('detections').order_by('results', direction=firestore.Query.DESCENDING).limit(1).stream()
        
        detection_data = None
        for doc in docs:
            detection_data = doc.to_dict()
        
        if detection_data:
            return jsonify({
                "detected_objects": detection_data.get('results', []),
                "image_url": detection_data.get('image_url', "")
            }), 200
        else:
            return jsonify({"message": "No detections found"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect():
    image_path = request.json.get('image_path')
    if not image_path:
        return jsonify({"error": "No image path provided"}), 400

    # Process the image and get the detection results
    detections, image_url, json_url = process_image(image_path)

    if detections is None:
        return jsonify({"error": "Error processing the image"}), 500

    # Return the results directly to the client (Thunkable) with JSON file link
    return jsonify({
        "detected_objects": detections,
        "image_url": image_url,
        "json_file_url": json_url
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
