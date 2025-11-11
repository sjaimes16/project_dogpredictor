import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# ============================================
# PASO 1: PREPARAR EL DATASET
# ============================================
# Necesitas organizar tus imágenes en carpetas:
# dataset/
#   train/
#     chihuahua/
#       img1.jpg, img2.jpg, ...
#     golden_retriever/
#       img1.jpg, img2.jpg, ...
#     labrador/
#       img1.jpg, img2.jpg, ...
#   validation/
#     chihuahua/
#     golden_retriever/
#     labrador/

# Configuración
IMG_SIZE = 224  # Tamaño estándar para modelos pre-entrenados
BATCH_SIZE = 32
EPOCHS = 20

# Rutas - ajusta según donde tengas las carpetas
train_dir = 'dataset/train'  # Carpeta con chihuahua/, golden_retriever/, husky/, desconocido/
validation_dir = 'dataset/test'  # Carpeta con las mismas subcarpetas para validación

# ============================================
# PASO 2: AUMENTO DE DATOS (DATA AUGMENTATION)
# ============================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Generadores de datos
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)
print(f"Clases encontradas: {train_generator.class_indices}")

# ============================================
# PASO 3: MODELO CON TRANSFER LEARNING
# ============================================
# Usamos MobileNetV2 pre-entrenado en ImageNet
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Congelamos las capas del modelo base
base_model.trainable = False

# Construimos nuestro modelo
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compilar
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# PASO 4: ENTRENAMIENTO
# ============================================
# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_dog_classifier.keras',
    monitor='val_accuracy',
    save_best_only=True
)

# Entrenar
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint]
)

# ============================================
# PASO 5: VISUALIZAR RESULTADOS
# ============================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ============================================
# PASO 7: EJEMPLOS DE USO
# ============================================
# Guardar el modelo
model.save('dog_breed_classifier_final.keras')

print("\n✅ Modelo entrenado exitosamente!")
print(f"📊 Precisión final: {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"💾 Modelo guardado en: dog_breed_classifier_final.keras")