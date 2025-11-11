import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tkinter import Tk, filedialog
import os

# ============================================
# CONFIGURACIÓN
# ============================================
IMG_SIZE = 224
MODEL_PATH = 'dog_breed_classifier_final.keras'
CONFIDENCE_THRESHOLD = 0.70  # 70% de confianza mínima

# Nombres de las clases (en orden alfabético, como fueron entrenadas)
CLASS_NAMES = ['beagle','bulldog_frances','chihuahua', 'desconocido', 'french_poodle', 'golden_retriever', 'husky', 'pastor_aleman', 'rottweiler', 'schnauzer']

# ============================================
# CARGAR EL MODELO
# ============================================
print("🔄 Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado exitosamente!\n")

# ============================================
# FUNCIÓN PARA SELECCIONAR IMAGEN
# ============================================
def select_image():
    """
    Abre un explorador de archivos para seleccionar una imagen
    
    Returns:
        str: ruta de la imagen seleccionada o None si se cancela
    """
    root = Tk()
    root.withdraw()  # Ocultar ventana principal de Tkinter
    root.wm_attributes('-topmost', 1)  # Poner ventana al frente
    
    print("\n📂 Abriendo explorador de archivos...")
    
    # Abrir diálogo de selección
    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen de perro",
        filetypes=[
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("Todos los archivos", "*.*")
        ],
        initialdir=os.getcwd()  # Iniciar en carpeta actual
    )
    
    root.destroy()
    
    if file_path:
        print(f"✅ Imagen seleccionada: {file_path}\n")
        return file_path
    else:
        print("❌ No se seleccionó ninguna imagen\n")
        return None

# ============================================
# FUNCIÓN PARA PREDECIR
# ============================================
def predict_dog_breed(image_path, show_plot=True):
    """
    Predice la raza de un perro desde una imagen
    
    Args:
        image_path: ruta de la imagen (ej: 'mi_perro.jpg')
        show_plot: si mostrar la imagen con el resultado
    
    Returns:
        tuple: (raza_predicha, confianza)
    """
    try:
        # Cargar y preprocesar la imagen
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Hacer predicción
        predictions = model.predict(img_array, verbose=0)
        max_confidence = np.max(predictions)
        predicted_class_idx = np.argmax(predictions)
        predicted_breed = CLASS_NAMES[predicted_class_idx]
        
        # Determinar si es desconocido por umbral de confianza
        if max_confidence < CONFIDENCE_THRESHOLD:
            final_result = "DESCONOCIDO"
            color = 'red'
            status = "⚠️"
        else:
            final_result = predicted_breed.upper()
            color = 'green' if predicted_breed != 'desconocido' else 'orange'
            status = "✅" if predicted_breed != 'desconocido' else "❓"
        
        # Mostrar resultados en consola
        print(f"\n{'='*50}")
        print(f"{status} RESULTADO DE LA PREDICCIÓN")
        print(f"{'='*50}")
        print(f"📸 Imagen: {image_path}")
        print(f"🐕 Raza predicha: {final_result}")
        print(f"📊 Confianza: {max_confidence*100:.2f}%")
        print(f"\n📋 Probabilidades por clase:")
        print(f"-"*50)
        
        # Ordenar probabilidades de mayor a menor
        sorted_indices = np.argsort(predictions[0])[::-1]
        for idx in sorted_indices:
            prob = predictions[0][idx]
            bar = '█' * int(prob * 40)
            print(f"  {CLASS_NAMES[idx]:20s} {prob*100:6.2f}% {bar}")
        
        print(f"{'='*50}\n")
        
        # Mostrar imagen con resultado
        if show_plot:
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f"{status} Predicción: {final_result}\nConfianza: {max_confidence*100:.2f}%", 
                     fontsize=16, color=color, weight='bold', pad=20)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return final_result, max_confidence
        
    except FileNotFoundError:
        print(f"❌ ERROR: No se encuentra la imagen '{image_path}'")
        return None, 0
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return None, 0

# ============================================
# FUNCIÓN PARA PREDECIR MÚLTIPLES IMÁGENES
# ============================================
def predict_multiple_images(image_paths):
    """
    Predice múltiples imágenes y muestra un resumen
    """
    results = []
    
    for img_path in image_paths:
        breed, confidence = predict_dog_breed(img_path, show_plot=False)
        if breed:
            results.append({
                'imagen': img_path,
                'raza': breed,
                'confianza': confidence
            })
    
    # Mostrar resumen
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN DE PREDICCIONES ({len(results)} imágenes)")
    print(f"{'='*60}")
    for r in results:
        status = "✅" if r['confianza'] >= CONFIDENCE_THRESHOLD else "⚠️"
        print(f"{status} {r['imagen']:30s} → {r['raza']:15s} ({r['confianza']*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return results

# ============================================
# EJEMPLOS DE USO
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🐕 CLASIFICADOR DE RAZAS DE PERROS")
    print("="*60)
    print("\nOpciones:")
    print("  1️⃣  Seleccionar imagen con explorador de archivos")
    print("  2️⃣  Escribir ruta manualmente")
    print("  3️⃣  Predecir múltiples imágenes")
    print("  4️⃣  Salir")
    print("="*60)
    
    while True:
        print("\n¿Qué deseas hacer?")
        opcion = input("Elige una opción (1-4): ").strip()
        
        if opcion == '1':
            # Usar explorador de archivos
            image_path = select_image()
            if image_path:
                predict_dog_breed(image_path)
                
        elif opcion == '2':
            # Escribir ruta manualmente
            print("\n📝 Ingresa la ruta de la imagen")
            print("Ejemplo: dataset/test/chihuahua/imagen1.jpg")
            image_path = input("\n🖼️  Ruta: ").strip()
            
            if image_path:
                predict_dog_breed(image_path)
                
        elif opcion == '3':
            # Predecir múltiples imágenes
            print("\n📂 Selecciona múltiples imágenes (una por una)")
            print("Presiona ENTER sin seleccionar para terminar\n")
            
            images = []
            while True:
                img = select_image()
                if img:
                    images.append(img)
                    continuar = input("¿Agregar otra imagen? (s/n): ").strip().lower()
                    if continuar != 's':
                        break
                else:
                    break
            
            if images:
                predict_multiple_images(images)
                
        elif opcion == '4' or opcion.lower() in ['salir', 'exit', 'quit']:
            print("\n👋 ¡Hasta luego!")
            break
            
        else:
            print("❌ Opción inválida. Elige 1, 2, 3 o 4")
        
        print("\n" + "-"*60)