# Utiliser l'image de base Python slim pour une taille de conteneur réduite
FROM python:3.10-slim

# Définir des variables d'environnement pour Python (optionnel mais bonne pratique)
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Installation de dépendances système nécessaires
# Pour les libs comme OpenCV (libgl1, libsm6, libxext6, libglib2.0-0)
# et gunicorn (bien que gunicorn soit maintenant inclus dans requirements.txt)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gunicorn \
    libgl1 \
    libsm6 \
    libxext6 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de l'application
# S'assurer que requirements.txt contient maintenant 'boto3'
COPY app.py requirements.txt /app/

# Installer les dépendances Python avec un délai d'attente (timeout) plus long.
# Ceci est crucial pour les gros packages comme PyTorch/NVIDIA qui peuvent dépasser
# le timeout par défaut (généralement 15 secondes).
RUN pip install --no-cache-dir --timeout 600 -r requirements.txt

# Exposer le port que l'application écoute
EXPOSE 8080

# Définir le point d'entrée pour Gunicorn
# Ceci démarrera le serveur Gunicorn avec l'application Flask définie dans app.py
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]