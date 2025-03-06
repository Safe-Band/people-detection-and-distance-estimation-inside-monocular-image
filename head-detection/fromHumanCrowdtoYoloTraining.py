import os
import shutil
import random

# Chemin vers le dossier contenant les images
image_folder = 'DATASET-CROWD-HUMAN/images/Imagess'
train_folder = 'DATASET-CROWD-HUMAN/images/train'
val_folder = 'DATASET-CROWD-HUMAN/images/val'

# Créer les dossiers d'entraînement et de validation s'ils n'existent pas
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Obtenir la liste des fichiers d'images
images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Mélanger les images de manière aléatoire
random.shuffle(images)

# Calculer le nombre d'images pour l'entraînement et la validation
train_size = int(0.9 * len(images))
val_size = len(images) - train_size

# Diviser les images en ensembles d'entraînement et de validation
train_images = images[:train_size]
val_images = images[train_size:]

# Copier les images dans les dossiers correspondants
for image in train_images:
    shutil.copy(os.path.join(image_folder, image), os.path.join(train_folder, image))

for image in val_images:
    shutil.copy(os.path.join(image_folder, image), os.path.join(val_folder, image))

print(f'Total images: {len(images)}')
print(f'Training images: {len(train_images)}')
print(f'Validation images: {len(val_images)}')
# Chemin vers le dossier contenant les labels
label_folder = 'DATASET-CROWD-HUMAN/labels/Annotations'
train_label_folder = 'DATASET-CROWD-HUMAN/labels/train'
val_label_folder = 'DATASET-CROWD-HUMAN/labels/val'

# Créer les dossiers d'entraînement et de validation pour les labels s'ils n'existent pas
os.makedirs(train_label_folder, exist_ok=True)
os.makedirs(val_label_folder, exist_ok=True)

# Copier les labels dans les dossiers correspondants
for image in train_images:
    label = os.path.splitext(image)[0] + '.txt'
    shutil.copy(os.path.join(label_folder, label), os.path.join(train_label_folder, label))

for image in val_images:
    label = os.path.splitext(image)[0] + '.txt'
    shutil.copy(os.path.join(label_folder, label), os.path.join(val_label_folder, label))