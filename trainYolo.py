from ultralytics import YOLO
import torch
print("CUDA available:", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
# Charger le modèle préexistant (ou choisir un modèle spécifique)
model = YOLO("yolo12n.pt").to(device)  # Charger le modèle pré-entraîné

# Définir le chemin du fichier YAML contenant les données (par exemple 'data.yaml')
data_path = 'YOLOtrain.yaml'

# Hyperparamètres pour l'entraînement
epochs = 20
batch_size = 16
imgsz = 540  # Utilise 'imgsz' au lieu de 'img_size'

# Lancer l'entraînement
model.train(data=data_path, epochs=epochs, batch=batch_size, imgsz=imgsz)
