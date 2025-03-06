from ultralytics import YOLO
import cv2

# Charger le modèle
model = YOLO("checkpoints/bestyolo11l.pt")  # Charger le modèle pré-entraîné

# Ouvrir le flux vidéo (0 pour la webcam ou chemin vers un fichier vidéo)
video_source = '63001-505518603_small.mp4' # Remplacez par le chemin vers un fichier vidéo si nécessaire
cap = cv2.VideoCapture(video_source)

# Vérifier si le flux vidéo s'ouvre correctement
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir le flux vidéo.")
    exit()

# Boucle pour lire les images du flux vidéo
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin du flux vidéo ou erreur de lecture.")
        break

    # Appliquer le modèle YOLO sur l'image
    results = model(frame)  # Faire les prédictions sur l'image

    # Annoter l'image avec les prédictions
    annotated_frame = results[0].plot()  # Dessiner les prédictions sur l'image

    # Afficher le résultat
    cv2.imshow("YOLOv11 - Détection en temps réel", annotated_frame)

    # Quitter la boucle si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
