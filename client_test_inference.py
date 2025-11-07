import requests
import base64
import json
import os
import sys
import uuid
import io

# Importations pour la visualisation
from PIL import Image
import cv2
import numpy as np

# --- CONFIGURATION (Veuillez vérifier) ---

# URL de l'API de votre Application Load Balancer (ALB)
API_URL = "http://CarDamageALB-1000365265.eu-central-1.elb.amazonaws.com/inference"

# Nom de l'image de voiture endommagée que vous allez utiliser pour le test.
IMAGE_FILE_PATH = "car-damage.jpg" # NOTE: Le nom de fichier utilisé dans le dernier test réussi

# Nom du fichier pour sauvegarder l'image avec les boîtes dessinées
OUTPUT_IMAGE_PATH = "car_damage_detected2.jpg"

# Dictionnaire de mappage (à adapter à votre modèle YOLO)
# Supposons que '0' signifie un dommage général
CLASS_MAPPING = {
    0: "General Damage (Small/Medium)",
    1: "Headlight Damage",
    # Ajoutez d'autres classes si votre modèle les supporte
}

# --- FONCTIONS UTILES ---

def encode_image_to_base64(image_path):
    """Lit un fichier image et le convertit en chaîne Base64."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except FileNotFoundError:
        print(f"ERREUR: Le fichier image '{image_path}' est introuvable.", file=sys.stderr)
        print("Veuillez vous assurer que vous avez placé une image nommée '{IMAGE_FILE_PATH}' dans ce répertoire.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de l'encodage de l'image: {e}", file=sys.stderr)
        sys.exit(1)

def estimate_cost(image_path, detection):
    """Estime le coût de réparation basé sur la taille de la détection et la classe."""
    # Logique d'estimation très simplifiée :
    # 1. Lire la taille de l'image pour normaliser les coordonnées (0-1000)
    img = cv2.imread(image_path)
    if img is None:
        return 0
        
    height, width, _ = img.shape
    
    # Coordonnées normalisées (0-1)
    box = detection['box']
    x_min, y_min, x_max, y_max = box[0] / width, box[1] / height, box[2] / width, box[3] / height
    
    # Calcul de la zone du dommage (normalisée)
    area = (x_max - x_min) * (y_max - y_min)
    
    # Coût de base par classe (exemple en EUR)
    base_cost = 0
    class_id = detection['class']
    
    if class_id == 0: # Dommage Général
        base_cost = 250 # Coût minimum de carrosserie
    elif class_id == 1: # Phare
        base_cost = 400 # Coût d'un phare neuf
    
    # Estimation finale : Coût de base + (Zone * Multiplicateur)
    # Plus la zone est grande, plus le coût est élevé.
    total_cost = base_cost + (area * 5000) 
    
    return round(total_cost, 2)


def draw_boxes(image_path, detections, output_path):
    """Charge l'image, dessine les boîtes englobantes et sauvegarde le résultat."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erreur: Impossible de charger l'image {image_path} pour la visualisation.", file=sys.stderr)
        return

    for det in detections:
        # Convertir les coordonnées float en int
        x_min, y_min, x_max, y_max = [int(c) for c in det['box']]
        class_id = det['class']
        confidence = det['confidence']
        
        # Mappage des classes et couleur
        label = CLASS_MAPPING.get(class_id, f"Unknown ({class_id})")
        color = (0, 255, 255) if class_id == 0 else (0, 0, 255) # Jaune pour 0, Rouge pour 1

        # Dessiner la boîte (rectangle)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        # Ajouter le texte (label + confiance + coût)
        cost = det.get('estimated_cost', 'N/A')
        text = f"{label} ({confidence*100:.1f}%) | Est. {cost}€"
        
        cv2.putText(img, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Sauvegarder l'image modifiée
    cv2.imwrite(output_path, img)
    print(f"\nImage visualisée sauvegardée sous: {output_path}")


def run_inference():
    """Exécute l'inférence en envoyant l'image encodée à l'API Fargate."""
    print(f"1. Tentative de lecture de l'image: {IMAGE_FILE_PATH}")
    base64_image = encode_image_to_base64(IMAGE_FILE_PATH)
    
    payload = {
        "image": base64_image,
        "image_id": f"test-yolov11-{os.path.basename(IMAGE_FILE_PATH)}-{uuid.uuid4()}" 
    }
    
    print(f"2. Envoi de la requête POST à l'API: {API_URL}")
    
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        
        response.raise_for_status()
        results = response.json()
        
        print("\n--- RÉSULTATS DE L'INFÉRENCE RÉUSSIS ---")
        
        detections = results.get('detections', [])
        
        total_estimated_cost = 0.0
        
        # 3. Traitement local : Calcul du coût et mise à jour de la détection
        if detections:
            for det in detections:
                estimated_cost = estimate_cost(IMAGE_FILE_PATH, det)
                det['estimated_cost'] = estimated_cost
                total_estimated_cost += estimated_cost

        print(f"Statut: {results.get('status')}")
        print(f"Image ID: {results.get('image_id')}")
        print(f"Détections trouvées: {len(detections)}")
        print("-" * 35)

        if len(detections) > 0:
            print(f"✅ COÛT TOTAL ESTIMÉ DE LA RÉPARATION : {total_estimated_cost:.2f} €")
            print("Détails des détections :")
            for i, det in enumerate(detections):
                # Utiliser la classe mappée pour un meilleur affichage
                label = CLASS_MAPPING.get(det['class'], f"Unknown ({det['class']})")
                print(f"  {i+1}. Type: {label}, Confiance: {det['confidence']:.4f}, Coût Est.: {det['estimated_cost']:.2f} €")
        else:
            print("Aucun dommage détecté sur cette image.")

        print("\n4. Visualisation : Dessin des boîtes sur l'image...")
        draw_boxes(IMAGE_FILE_PATH, detections, OUTPUT_IMAGE_PATH)
        
        print("\n5. Vérification DynamoDB (les résultats bruts sont stockés).")

    except requests.exceptions.HTTPError as errh:
        print(f"Erreur HTTP (API Error): {errh}", file=sys.stderr)
        # S'assurer d'imprimer le corps de la réponse en cas d'erreur 4xx/5xx
        if response is not None and response.text:
             print(f"Réponse de l'API: {response.text}", file=sys.stderr)
    except requests.exceptions.RequestException as err:
        print(f"Erreur inattendue lors de la requête: {err}", file=sys.stderr)


# --- EXÉCUTION ---
if __name__ == "__main__":
    
    # Vérification des dépendances
    try:
        import requests
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"Le module '{e.name}' est manquant. Installez-le avec: pip install requests opencv-python Pillow", file=sys.stderr)
        sys.exit(1)
        
    run_inference()
