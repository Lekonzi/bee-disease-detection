import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import load_img, img_to_array

# charger modèle
model = tf.keras.models.load_model("bee_disease_model.h5")

# noms des classes
class_names = [
    'Varroa,_Small_Hive_Beetles',
    'ant_problems',
    'few_varrao,_hive_beetles',
    'healthy',
    'hive_being_robbed',
    'missing_queen'
]

# chemin image
img_path = "test/image_test.png"

# charger image
img = load_img(
    img_path,
    target_size=(224, 224)
)

# convertir image
img_array = img_to_array(img)

# normalisation
img_array = img_array / 255.0

# ajouter dimension batch
img_array = np.expand_dims(img_array, axis=0)

# prédiction
prediction = model.predict(img_array)

# résultat
predicted_class = class_names[np.argmax(prediction)]

confidence = np.max(prediction) * 100

# afficher résultat terminal
print("Classe prédite :", predicted_class)
print("Confiance :", confidence, "%")


# AFFICHER IMAGE + PREDICTION

plt.figure(figsize=(6,6))

# afficher image
plt.imshow(img)

# titre
plt.title(
    f"Prediction : {predicted_class}\nConfidence : {confidence:.2f}%"
)

# enlever axes
plt.axis("off")

# afficher fenêtre
plt.show()