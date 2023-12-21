from ultralytics import YOLO
import cv2
import numpy as np

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
cap = cv2.VideoCapture('../VID_2.mp4')

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

        # Seguir vehículos
        ids_seguidos = seguidor_vehiculos.update(np.asarray(detecciones_))

        # Detectar placas de matrícula
        placas_matricula = detector_placas(fotograma)[0]
       
        for placa_matricula in placas_matricula.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = placa_matricula

            # Asignar la placa de matrícula al vehículo
            xcar1, ycar1, xcar2, ycar2, id_vehiculo = get_car(placa_matricula, ids_seguidos)
            
            if id_vehiculo != -1:
                # Recortar la placa de matrícula
                placa_matricula_recortada = fotograma[int(y1):int(y2), int(x1): int(x2), :]

                cv2.rectangle(fotograma, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 200), 2)
                # Procesar la placa de matrícula
                # placa_matricula_gris = cv2.cvtColor(placa_matricula_recortada, cv2.COLOR_BGR2GRAY)
                # _, placa_matricula_umbralizada = cv2.threshold(placa_matricula_gris, 64, 255, cv2.THRESH_BINARY_INV)
                placa_matricula_umbralizada= procesar_imagen(placa_matricula_recortada)
                # Leer el número de la placa de matrícula
                texto_placa_matricula, puntaje_texto_placa_matricula = read_license_plate(placa_matricula_umbralizada)
                # Guardar la imagen de la placa
                nombre_archivo = f'placa_{id_vehiculo}_{numero_fotograma}_b.png'
                #cv2.imwrite(nombre_archivo, placa_matricula_umbralizada)
                # Agregar el texto sobre el rectángulo de la placa
                if puntaje_texto_placa_matricula is None:
                    puntaje_texto_placa_matricula=0
              # Actualizar mejor reconocimiento y mejor puntaje
                print("===========>" + str(id_vehiculo))
                id_vehiculo1 = int(id_vehiculo)

                if puntaje_texto_placa_matricula > mejor_puntaje[id_vehiculo1]:
                    mejor_reconocimiento[id_vehiculo1] = texto_placa_matricula
                    mejor_puntaje[id_vehiculo1] = puntaje_texto_placa_matricula
               
                cv2.putText(fotograma, mejor_reconocimiento[id_vehiculo1], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

                if texto_placa_matricula is not None:
                    resultados[numero_fotograma][id_vehiculo] = {'vehiculo': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                                 'placa_matricula': {'bbox': [x1, y1, x2, y2],
                                                                                     'texto': texto_placa_matricula,
                                                                                     'bbox_puntaje': score,
                                                                                     'texto_puntaje': puntaje_texto_placa_matricula}}

        # Escribir los resultados
        write_csv(resultados, './test5.csv')        
        # Mostrar el fotograma con los rectángulos dibujados
        cv2.imshow('Vehículos Detectados', fotograma)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
