import tensorflow as tf
import matplotlib.pyplot as plt

# paramètres
img_size = (224, 224)
batch_size = 32

# chargement des données
train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# sauvegarder les classes
class_names = train_data.class_names
num_classes = len(class_names)

print("Classes :", class_names)

# normalisation
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_data = train_data.map(
    lambda x, y: (normalization_layer(x), y)
)

val_data = val_data.map(
    lambda x, y: (normalization_layer(x), y)
)

# modèle CNN
model = tf.keras.models.Sequential([

    tf.keras.Input(shape=(224,224,3)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# entraînement
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=3

)

# évaluation
loss, acc = model.evaluate(val_data)

print("Accuracy :", acc)

# AFFICHAGE DES 2 GRAPHIQUES


fig, axes = plt.subplots(1, 2, figsize=(12,5))


# Accuracy

axes[0].plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

axes[0].plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

axes[0].set_title('Accuracy du modèle')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')

axes[0].legend()

# Loss

axes[1].plot(
    history.history['loss'],
    label='Train Loss'
)

axes[1].plot(
    history.history['val_loss'],
    label='Validation Loss'
)

axes[1].set_title('Loss du modèle')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')

axes[1].legend()

# affichage propre
plt.tight_layout()

# afficher les 2 graphes dans UNE fenêtre
plt.show()

# SAUVEGARDE DU MODÈLE

model.save("bee_disease_model.h5")

print("Modèle sauvegardé ")