import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model("model.h5")

# Capturar un video
cap = cv2.VideoCapture("video.mp4")

while True:
  # Leer un frame del video
  ret, frame = cap.read()

  # Detectar las personas en el frame
  personas = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

  # Iterar sobre las personas detectadas
  for (x, y, w, h) in personas:
    # Obtener la región de interés
    region_de_interes = frame[y:y + h, x:x + w]

    # Detectar armas de fuego en la región de interés
    armas_de_fuego = model.predict(region_de_interes)

    # Si se detecta un arma de fuego, enviar una notificación
    if armas_de_fuego > 0.5:
      notificacion = "Se ha detectado una persona con un arma de fuego en las coordenadas (x, y)"
      print(notificacion)

  # Mostrar el frame en pantalla
  cv2.imshow("Video", frame)
  if cv2.waitKey(1) == 27:
    break

# Cerrar la captura de video
cap.release()

# Destruir todas las ventanas abiertas
cv2.destroyAllWindows()