# --- Standard Flask and Libraries Imports ---
# ATTENTION: make_response est importé pour gérer les requêtes OPTIONS du CORS
from flask import Flask, request, jsonify, make_response 
import os
import sys
import datetime
import uuid
import io
import base64
from PIL import Image

# Assurez-vous d'importer toutes vos bibliothèques ML nécessaires
from ultralytics import YOLO 
import boto3
import json

# --- Configuration et Variables Globales ---
# Les valeurs par défaut sont utilisées pour les tests locaux
S3_BUCKET_MODELS = os.environ.get('S3_BUCKET_MODELS', 'car-damage-3051-models')
DYNAMODB_TABLE_NAME = os.environ.get('DYNAMODB_TABLE_NAME', 'CarDamageInferenceResults')
MODEL_PATH = '/tmp/best.pt' 

# NOUVELLE CLÉ S3: Chemin complet vers le fichier dans le bucket
# Le chemin est: artifacts/yolov11/best.pt (assurez-vous que la casse est correcte dans S3)
S3_MODEL_KEY = 'artifacts/yolov11/best.pt' 

# Variable globale pour stocker le modèle chargé
model = None 

# --- MODEL INITIALIZATION SECTION (RUNS ONCE PER WORKER PROCESS STARTUP) ---
def load_model_sync():
    """Télécharge et charge le modèle ML.
    Ceci est appelé au niveau du module et s'exécute une seule fois par processus Gunicorn worker.
    """
    try:
        s3 = boto3.client('s3')
        # La ligne de print utilise maintenant la clé S3 correcte
        print(f"Téléchargement du modèle depuis s3://{S3_BUCKET_MODELS}/{S3_MODEL_KEY} vers {MODEL_PATH}")
        
        # Utilisation du chemin S3 correct (clé)
        s3.download_file(S3_BUCKET_MODELS, S3_MODEL_KEY, MODEL_PATH)

        # Charger le modèle YOLO
        loaded_model = YOLO(MODEL_PATH) 
        print("Modèle chargé avec succès.")
        return loaded_model
    except Exception as e:
        # Erreur critique: l'impression sur stderr est essentielle pour le débogage
        print(f"FATAL: ERREUR LORS DU CHARGEMENT DU MODÈLE: {e}", file=sys.stderr)
        return None

# Appel direct de la fonction de chargement au niveau du module.
model = load_model_sync() 

# --- Flask App Definition ---
app = Flask(__name__)

# --- GESTION CORS (Correction de l'erreur 'Failed to fetch' liée aux Headers) ---

def add_cors_headers(response):
    """Ajoute les en-têtes CORS pour permettre l'appel depuis l'application web."""
    # Le '*' autorise tous les domaines (y compris 127.0.0.1) à appeler cette API
    response.headers['Access-Control-Allow-Origin'] = '*' 
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# Cette fonction s'exécute après chaque requête et ajoute les headers CORS
app.after_request(add_cors_headers)

# Endpoint pour les requêtes de pré-vérification OPTIONS (CORS)
@app.route('/inference', methods=['OPTIONS'])
def handle_options():
    """Gère les requêtes OPTIONS (preflight CORS)"""
    # Retourne une réponse vide avec le statut 200, mais les headers CORS sont ajoutés via app.after_request
    return make_response('', 200)

# --- Endpoints de l'API ---

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de santé pour l'ALB."""
    # Le modèle est utilisé comme indicateur de santé
    if model is not None:
        return jsonify({"status": "OK", "model_loaded": True}), 200
    # Retourne 503 si le modèle n'a pas pu être chargé, indiquant un service non sain à l'ALB
    return jsonify({"status": "UNHEALTHY", "model_loaded": False, "details": "Model failed to load."}), 503 

@app.route('/inference', methods=['POST'])
def inference_handler():
    """Endpoint principal pour exécuter l'inférence."""
    if model is None:
        # Unlikely to hit this if health check works, but good practice
        return jsonify({"error": "Model not loaded. Service is UNHEALTHY."}), 503

    try:
        data = request.json
        image_base64 = data.get('image')
        
        # Validation d'entrée
        if not image_base64:
             return jsonify({"error": "Missing 'image' (base64) field in request body."}), 400
             
        # Générer l'ID pour la clé primaire de DynamoDB
        assessment_id = str(uuid.uuid4())
        
        # 1. Décoder l'image
        # Note: Le client JS devrait envoyer la partie pure base64 (sans 'data:image/...')
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))

        # 2. Exécuter l'inférence
        results = model(image)
        
        detections = []
        for r in results:
            boxes = r.boxes.xyxy.tolist()
            classes = r.boxes.cls.tolist()
            confidences = r.boxes.conf.tolist()
            
            for box, cls, conf in zip(boxes, classes, confidences):
                detections.append({
                    # Arrondir les valeurs pour un JSON propre et compatible DynamoDB
                    "box": [round(c, 2) for c in box],
                    "class": int(cls),
                    "confidence": round(conf, 4)
                })

        # 3. Enregistrer dans DynamoDB
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(DYNAMODB_TABLE_NAME)
        
        table.put_item(
            Item={
                # CHANGEMENT CLÉ : Utiliser "assessmentId" pour correspondre au schéma de la table
                'assessmentId': assessment_id, 
                'timestamp': str(datetime.datetime.now()),
                # Stocker les résultats en string JSON pour éviter les problèmes de type
                'results_json': json.dumps(detections) 
            }
        )
        
        return jsonify({
            "assessmentId": assessment_id, # Renvoyer le nouvel ID généré
            "status": "Inference Complete and Result Saved",
            "detection_count": len(detections),
            "detections": detections
        }), 200

    except Exception as e:
        # Loggez l'erreur pour le débogage CloudWatch
        print(f"Erreur lors de l'inférence: {e}", file=sys.stderr)
        return jsonify({"error": f"Internal Server Error during inference: {str(e)}"}), 500

# --- Lancement du Serveur ---

if __name__ == '__main__':
    # Ceci est utilisé seulement pour le test local direct (sans gunicorn)
    app.run(host='0.0.0.0', port=8080)
