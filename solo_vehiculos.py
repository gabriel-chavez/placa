from ultralytics import YOLO
import cv2
import numpy as np
import random

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv, procesar_imagen

resultados = {}

seguidor_vehiculos = Sort()
vehiculos = [2, 3, 5, 7]

# Cargar modelos
modelo_coco = YOLO('yolov8n.pt')
detector_placas = YOLO('license_plate_detector.pt')

# Cargar video
#cap = cv2.VideoCapture('./20231219_164345.mp4')
cap = cv2.VideoCapture('../VID_20240106_115946.mp4')
#cap = cv2.VideoCapture(0) 
# Definir las nuevas dimensiones (ancho, alto) para la ventana redimensionada   
nuevas_dimensiones_ventana = (540, 380)  # Puedes ajustar estas dimensiones según tus necesidades

# Crear la ventana de salida con el nombre 'Vehículos Detectados'
cv2.namedWindow('Vehículos Detectados', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Vehículos Detectados', nuevas_dimensiones_ventana[0], nuevas_dimensiones_ventana[1])
# Inicializar arrays de NumPy
mejor_reconocimiento = np.empty(100000, dtype=object)  # Suponiendo un límite de 1000 vehículos
mejor_puntaje = np.zeros(100000)
# Leer fotogramas
numero_fotograma = -1
ret = True
while ret:
    numero_fotograma += 1
    ret, fotograma = cap.read()
    if ret:
        # # Redimensionar el fotograma con especificación de método de interpolación
        #fotograma = cv2.resize(fotograma, nuevas_dimensiones_ventana, interpolation=cv2.INTER_NEAREST)

        resultados[numero_fotograma] = {}
        # Detectar vehículos
        detecciones = modelo_coco(fotograma)[0]
        detecciones_ = []
        
        
        for deteccion in detecciones.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = deteccion
            if int(class_id) in vehiculos:
                detecciones_.append([x1, y1, x2, y2, score])

                # Dibujar un rectángulo alrededor del vehículo con un color blanco y grosor 3
                cv2.rectangle(fotograma, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)

        # # Seguir vehículos
        # try:
        #     ids_seguidos = seguidor_vehiculos.update(np.asarray(detecciones_))
        # except:
        #     ids_seguidos=0

        vehiculo_recortado = fotograma[int(y1):int(y2), int(x1): int(x2), :]
        nombre_archivo2 = f'v_{numero_fotograma}_{random.randrange(10000)}_1.png'
        cv2.imwrite(nombre_archivo2, vehiculo_recortado)
        # Detectar placas de matrícula
        
        # Escribir los resultados
        #write_csv(resultados, './test5.csv')        
        # Mostrar el fotograma con los rectángulos dibujados
        cv2.imshow('Vehículos Detectados', fotograma)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
