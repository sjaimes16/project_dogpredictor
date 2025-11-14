🐶 Dog Predictor
Modelo de IA para clasificación de imágenes con TensorFlow

Este proyecto implementa un modelo de inteligencia artificial capaz de analizar imágenes y realizar predicciones relacionadas con perros.
El repositorio es público, por lo que cualquier persona puede descargarlo y usarlo, incluso sin conocimientos de programación.

📦 Contenido del repositorio
📁 dataset/           # Imágenes utilizadas por el modelo
📄 otro_modelo.py     # Genera o entrena el modelo (.keras)
📄 dog_predictor.py   # Carga el modelo y realiza predicciones
📄 README.md          # Documentación del proyecto

📥 Instalación y descarga
🟢 Opción para principiantes: Descargar ZIP

Haz clic en el botón Code (arriba a la derecha).

Selecciona Download ZIP.

Extrae/descomprime el archivo en tu computadora.

✔ Esta es la forma más fácil si no tienes experiencia con Git o programación.

🟣 Opción para usuarios con Git (clonado)
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git

🛠️ Requisitos del sistema

Necesitas instalar:

✔ Python 3.10 o superior

Descargar desde: https://www.python.org/downloads/

Durante la instalación marca:

✔ Add Python to PATH

✔ Instalar dependencias

Abre una terminal dentro de la carpeta del proyecto y ejecuta:

pip install numpy tensorflow matplotlib tkinter

🚀 Cómo utilizar el proyecto

Este proyecto funciona en dos pasos principales:

1️⃣ Generar el archivo del modelo (.keras)

Ejecuta:

python otro_modelo.py

Este script creará un archivo .keras que contiene el modelo entrenado.
Debe ejecutarse primero.

2️⃣ Ejecutar el predictor

Una vez generado el .keras, ejecutar:

python dog_predictor.py

Este script carga el modelo y permite realizar predicciones usando imágenes.

🧪 Ejemplo de uso del predictor

El archivo dog_predictor.py funciona mediante un menú interactivo con 4 opciones, por lo que no necesitas escribir comandos complicados.
Solo debes ejecutarlo así:

python dog_predictor.py

Una vez iniciado, verás un menú como este:

¿Qué deseas hacer?
1. Seleccionar una imagen usando el explorador de archivos
2. Escribir la ruta de una imagen manualmente
3. Seleccionar varias imágenes (una por una)
4. Salir

A continuación, se describe cada opción:

🔹 Opción 1 – Seleccionar imagen con explorador

Abre una ventana de explorador de archivos para elegir una imagen.
El programa cargará la imagen y mostrará la predicción.

🔹 Opción 2 – Ingresar la ruta manualmente

Puedes escribir la ruta donde está tu imagen.
Ejemplo:

dataset/test/chihuahua/imagen1.jpg


El programa cargará esa imagen y realizará la predicción.

🔹 Opción 3 – Predecir múltiples imágenes

Permite seleccionar varias imágenes, una por una, usando el explorador.
Después de cada imagen, podrás decidir si quieres agregar otra:

¿Agregar otra imagen? (s/n)

Cuando termines, el programa procesará todas las imágenes y mostrará las predicciones.

🔹 Opción 4 – Salir

Finaliza el programa y cierra el menú.

❓ Preguntas frecuentes (FAQ)
✔ ¿Necesito saber programar?

No. Solo sigue las instrucciones de instalación y ejecución.

✔ ¿Puedo usar mis propias imágenes?

Sí, solo reemplaza la imagen que usa el script o modifícalo para cargar otras.

✔ ¿Funciona en Windows, Mac y Linux?

Sí, mientras Python esté instalado.

📄 Licencia

Este proyecto utiliza una licencia MIT, lo que permite usarlo, modificarlo y distribuirlo libremente.
