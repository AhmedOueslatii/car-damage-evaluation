{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "96ef61d8",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'sagemaker'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 3\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;66;03m# Cellule 1 : Configuration et Définition des Chemins\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;66;03m# Rôle et Session déjà définis (Assurez-vous que le rôle IAM est un 'SageMaker - Execution')\u001b[39;00m\n\u001b[1;32m----> 3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01msagemaker\u001b[39;00m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01msagemaker\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mpytorch\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mimport\u001b[39;00m PyTorch\n\u001b[0;32m      6\u001b[0m sagemaker_session \u001b[38;5;241m=\u001b[39m sagemaker\u001b[38;5;241m.\u001b[39mSession()\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'sagemaker'"
     ]
    }
   ],
   "source": [
    "# Cellule 1 : Configuration et Définition des Chemins\n",
    "# Rôle et Session déjà définis (Assurez-vous que le rôle IAM est un 'SageMaker - Execution')\n",
    "import sagemaker\n",
    "from sagemaker.pytorch import PyTorch\n",
    "\n",
    "sagemaker_session = sagemaker.Session()\n",
    "role = sagemaker.get_execution_role() \n",
    "\n",
    "# Chemins S3 (Données sont stockées ici)\n",
    "S3_DATA_PATH = 's3://car-damage-3051-images-raw/roboflow-car-damage-yolov11/' \n",
    "S3_OUTPUT_PATH = f's3://car-damage-3051-images-raw/output/yolo11-training/' # Changement du chemin de sortie pour YOLO11\n",
    "\n",
    "inputs = {'training': S3_DATA_PATH}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "49f6f8d0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "py310_env",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
