from flask import Flask, request, jsonify, render_template
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from GPDCNN import GPDCNN
from queue import Queue
import threading
import os
import uuid
import numpy as np
import json

app = Flask(__name__)

# Custom JSON Encoder
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

# Database setup
DATABASE_URL = "mysql+pymysql://tl_bot:1234567890@localhost:8889/plant_diagnoses"  # Change to your actual database URL
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Define Disease model
class Disease(Base):
    __tablename__ = 'disease_on_leaf'
    id = Column(Integer, primary_key=True)
    id = Column(Integer, primary_key=True)
    disease_code = Column(String(30))  # VARCHAR(30)
    disease_en = Column(String(30))    # VARCHAR(30)
    disease_km = Column(String(30))    # VARCHAR(30)
    cure = Column(Text)                # TEXT
    symtom = Column(Text)              # TEXT
    reference = Column(Text)           # TEXT
    status = Column(Integer)           # int(11)

# Ensure tables are created
Base.metadata.create_all(engine)

class ImageProcessor:
    def __init__(self):
        self.predictor = GPDCNN()
        self.image_queue = Queue()
        self.processing_thread = threading.Thread(target=self.process_images, daemon=True)
        self.processing_thread.start()
        self.results = {}

    def add_image(self, image_path):
        image_id = str(uuid.uuid4())
        self.image_queue.put((image_id, image_path))
        self.results[image_id] = {"status": "processing"}
        print(f"Added image with ID: {image_id}")
        return image_id

    def process_images(self):
        while True:
            image_id, image_path = self.image_queue.get()
            try:
                result = self.predictor.predict(image_path)
                if result is None:
                    raise ValueError("Prediction returned None")
                
                predicted_disease, confidence = result
                print(f"Processed image: {image_path}")
                print(f"Predicted Disease: {predicted_disease}")
                print(f"Confidence: {confidence:.2f}")
                session = Session()
                # Query database for the disease code
                disease_info = session.query(Disease).filter_by(disease_code=predicted_disease).first()

                if disease_info:
                    # Print out the disease details
                    print("Retrieved Disease Details:")
                    print(f"disease_km: {disease_info.disease_km}")
                    print(f"cure: {disease_info.cure}")
                    print(f"symtom: {disease_info.symtom}")
                    print(f"reference: {disease_info.reference}")
                    
                    self.results[image_id] = {
                        "status": "completed",
                        "result": {
                            "predicted_disease": predicted_disease,
                            "confidence": confidence,
                            "details": {
                                "disease_km": disease_info.disease_km,
                                "cure": disease_info.cure,
                                "symtom": disease_info.symtom,
                                "reference": disease_info.reference,
                            }
                        }
                    }
                else:
                    print("Disease details not found in the database.")
                    self.results[image_id] = {"status": "error", "message": "Disease details not found"}
                
                session.close()
            except FileNotFoundError:
                print(f"Error: Image file not found at {image_path}")
                self.results[image_id] = {"status": "error", "message": "File not found"}
            except ValueError as ve:
                print(f"Error processing image {image_path}: {str(ve)}")
                self.results[image_id] = {"status": "error", "message": str(ve)}
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                self.results[image_id] = {"status": "error", "message": str(e)}
            finally:
                self.image_queue.task_done()

processor = ImageProcessor()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = file.filename
        file_path = os.path.join(os.getcwd(), 'uploads', filename)
        file.save(file_path)
        image_id = processor.add_image(file_path)
        return jsonify({'message': 'File uploaded successfully', 'image_id': image_id}), 200

@app.route('/result/<image_id>', methods=['GET'])
def get_result(image_id):
    result = processor.results.get(image_id)
    if result is None:
        return jsonify({'error': 'Image not found'}), 404
    
    if result['status'] == 'completed':
        return jsonify(result)
    elif result['status'] == 'processing':
        return jsonify({'status': 'processing'})
    else:
        return jsonify(result)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

