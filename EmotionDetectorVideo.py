from tkinter import *
import cv2
import imutils
import os
import numpy as np
import time
from PIL import Image, ImageDraw, ImageTk  
from threading import Thread
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Variables para calcular FPS
time_actualframe = time_prevframe = time.time()

# Tipos de emociones del detector
classes = ['angry','disgust','fear','happy','neutral','sad', 'surprise']


# Ruta donde se guardarán las imágenes de los usuarios detectados
save_path = "asistencia/"
photo_taken = False

# Si la ruta no existe, la crea
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Carpeta '{save_path}' creada.")

# Cargamos el modelo de detección de rostros
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carga el detector de clasificación de emociones
emotionModel = load_model("modelFEC.h5")

# Se crea la captura de video
cam = None

# Tamaño de la ventana
window_width = 640
window_height = 480

# Inicializar la ventana tkinter
root = Tk()
root.title("Detector de emociones")

# Crear una etiqueta para mostrar el video
video_label = Label(root, width=window_width, height=window_height)
video_label.pack()


# Funcion para capturar y guardar la foto
def capture_and_save_photo():
    global cam, photos_captured, photo_taken
    ret, frame = cam.read()
    if ret:
        photo_path = os.path.join(save_path, f"photo_{photos_captured}.png")
        cv2.imwrite(photo_path, frame)
        print(f"Foto guardada en: {photo_path}")
        photos_captured += 1
        load_present_students_images()
        photo_taken = True 


# Rutas de las imágenes del botón
camera_of_image_path = "frames/iconos/camara_cerrada.jpg"
camera_on_image_path = "frames/iconos/camara_abierta.jpg"

camera_of_image = ImageTk.PhotoImage(Image.open(camera_of_image_path))
camera_on_image = ImageTk.PhotoImage(Image.open(camera_on_image_path))

# Variable para almacenar el estado actual del botón
camera_button_state = False
photos_captured = 0

# Establecer el icono dependiendo del estado de la camara
def switch_camera_image():
    global camera_button_state
    if camera_button_state == False:
        camera_button.config(image=camera_of_image)
    else:
        camera_button.config(image=camera_on_image)

# Función para cambiar la imagen del botón
def switch_camera_state():
    global camera_button_state
    if camera_button_state:
        stop_capture()
    else:
        start_capture()

# Función para encender la cámara
def start_capture():
    global cam, camera_button_state
    if not camera_button_state:  # Verificar si la cámara está apagada
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera_button_state = True
        switch_camera_image()
        thread = Thread(target=capture_video)
        thread.start()

# Función para apagar la cámara
def stop_capture():
    global cam, camera_button_state
    if cam is not None and camera_button_state:
        cam.release()
        camera_button_state = False
        cam = None
        switch_camera_image()

# Dibujar recadro donde se muestra la camara
camera_frame = Frame(root)
camera_frame.pack(side=RIGHT)

# Frame para dibujar estudiante presentes
sidebar_frame = Frame(root, bg="#000", width=200)
sidebar_frame.pack(side=LEFT, fill=Y)

# Establecer un titulo de los estudiantes presentes
sidebar_title_label = Label(sidebar_frame, text="Estudiantes presentes", bg="#000", fg="#fff", font=("Arial", 14, "bold"))
sidebar_title_label.pack(pady=10, padx=10)

# Crear un botón para controlar la cámara
camera_button = Button(camera_frame, image=camera_of_image, command=switch_camera_state, borderwidth=0, cursor="hand2")
camera_button.pack(padx=(0, 10))  # Ajustar el padding en x (horizontal) y y (vertical)

# Cargar la imagen del estudiante presente
def load_present_students_images():

    # Obtener la lista de imágenes de los estudiantes presentes en la carpeta
    present_students_images = [file for file in os.listdir(save_path) if file.endswith(".png")]

    # Mostrar las imágenes en el sidebar
    for image_name in present_students_images:
        image_path = os.path.join(save_path, image_name)
        image = Image.open(image_path)
        image.thumbnail((100, 100))  # Redimensionar la imagen para que quepa en el sidebar
        photo = ImageTk.PhotoImage(image)

        # Crear una etiqueta para mostrar la imagen en el sidebar
        image_label = Label(sidebar_frame, image=photo)
        image_label.image = photo  # Mantener una referencia al objeto de imagen para evitar que se elimine
        image_label.pack(pady=5)

# Función para mostrar un fondo negro en la ventana tkinter
def display_black_screen():
    black_screen = Image.new("RGBA", (window_width, window_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(black_screen)
    border_width = 5  # Ancho del borde
    draw.rectangle([(border_width, border_width), (window_width - border_width, window_height - border_width)], outline="black", width=border_width)
    print("el bicho velde")
    # Convertir la imagen a formato TKinter
    black_screen_tk = ImageTk.PhotoImage(black_screen)
    
    # Mostrar la imagen en el widget
    video_label.config(image=black_screen_tk)
    video_label.image = black_screen_tk

# Función para capturar el video y mostrar las emociones detectadas
def capture_video():
    global cam, camera_button_state, time_prevframe, photo_taken
    while camera_button_state:
        # Se toma un frame de la cámara y se redimensiona
        ret, frame = cam.read()
        if not ret:
            print("No se pudo leer el fotograma de la cámara")
            break

        frame = imutils.resize(frame, width=window_width)

        # Predecir la emocion del usuario actual
        locs, preds = predict_emotion(frame, faceNet, emotionModel)

        # Si la foto no esta tomada, tomar la foto
        if not photo_taken:
            capture_and_save_photo()

            # Verificar si se han detectado rostros antes de procesar las emociones
        if locs is not None and preds is not None:
                # Para cada hallazgo se dibuja en la imagen el bounding box y la clase
                for (box, pred) in zip(locs, preds):
                    (Xi, Yi, Xf, Yf) = box
                    (angry, disgust ,fear, happy, neutral, sad, surprise) = pred

                    label = ''
                    # Se agrega la probabilidad en el label de la imagen
                    label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(angry, disgust, fear, happy, neutral, sad, surprise) * 100)

                    cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 3)
                    cv2.putText(frame, label, (Xi + 5, Yi - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        time_actualframe = time.time()

        if time_actualframe > time_prevframe:
            fps = 1 / (time_actualframe - time_prevframe)

        time_prevframe = time_actualframe

        cv2.putText(frame, str(int(fps)) + " FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

        # Mostrar el frame en la ventana tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        video_label.config(image=frame)
        video_label.image = frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Función para predecir las emociones en los rostros detectados
def predict_emotion(frame, faceNet, emotionModel):
    global detected_users, photo_taken
    # Construye un blob de la imagen
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Realiza las detecciones de rostros a partir de la imagen
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Inicializa las listas de ubicaciones y predicciones
    locs = []
    preds = []

    # Recorre cada una de las detecciones
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.4:
            # Toma el bounding box de la detección escalado
            # de acuerdo a las dimensiones de la imagen
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            # Valida las dimensiones del bounding box
            if Xi < 0: Xi = 0
            if Yi < 0: Yi = 0

            # Se extrae el rostro y se convierte BGR a GRAY
            # Finalmente se escala a 224x244
            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face2 = img_to_array(face)
            face2 = np.expand_dims(face2, axis=0)

            # Guardar ubicación y predicciones
            locs.append((Xi, Yi, Xf, Yf))
            preds.append(emotionModel.predict(face2)[0])

    # Devuelve las ubicaciones y predicciones (o None si no se detectaron rostros)
    return (locs, preds) if len(locs) > 0 else (None, None)
                # Detener la captura de la cámara después de guardar la foto
            
# Mostrar fondo negro inicialmente
display_black_screen()

# Bucle principal para mostrar la interfaz gráfica
root.mainloop()
