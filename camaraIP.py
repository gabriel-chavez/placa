import cv2
import requests
import numpy as np


from ultralytics import YOLO

from sort.sort import Sort
from util import get_car, leer_placa, write_csv, procesar_imagen_color,procesar_imagen_ajustes, mejor_texto_reconocido


def show_dahua_live_video(ip_address, username, password):
    #Cargar modelos
    modelo_coco = YOLO('yolov8n.pt')
    detector_placas = YOLO('license_plate_detector.pt')


    resultados = {}
    seguidor_vehiculos = Sort()
    vehiculos = [2, 3, 5, 7]


    # # Cambiar el método de lectura
    # cap = cv2.VideoCapture('../VID_20240106_115946.mp4')
    # #cap = cv2.VideoCapture(0) 



    nuevas_dimensiones_ventana = (540, 380)  # Puedes ajustar estas dimensiones según tus necesidades#
    cv2.namedWindow('Vehículos Detectados', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Vehículos Detectados', nuevas_dimensiones_ventana[0], nuevas_dimensiones_ventana[1])

    # Inicializar arrays de NumPy
    mejor_reconocimiento = np.empty(100000, dtype=object)
    mejor_puntaje = np.zeros(100000)

    # Leer fotogramas
    numero_fotograma = -1
    ret = True
    # URL del flujo de video de la cámara PTZ IP de Dahua
    url = f"rtsp://{username}:{password}@{ip_address}/cam/realmonitor?channel=1&subtype=0"

    # Crear un objeto de captura de video
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Error al abrir el flujo de video.")
        return

    while ret:
        numero_fotograma += 1
        ret, fotograma = cap.read()
        if ret:
            resultados[numero_fotograma] = {}
        
            # Detectar vehículos
            detecciones = modelo_coco(fotograma)[0]
            detecciones_ = []

            for deteccion in detecciones.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = deteccion
                if int(class_id) in vehiculos:
                    detecciones_.append([x1, y1, x2, y2, score])
                    cv2.rectangle(fotograma, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)

            # Seguir vehículos
            try:
                ids_seguidos = seguidor_vehiculos.update(np.asarray(detecciones_))
            except:
                ids_seguidos = 0
            if (ids_seguidos != 0).any():
                # Detectar placas de matrícula
                placas_matricula = detector_placas(fotograma)[0]
                #for placa_matricula in placas_matricula.boxes.data.tolist():
                if placas_matricula.boxes.data.tolist():
                    placa_matricula = placas_matricula.boxes.data.tolist()[0]
                    
                    x1, y1, x2, y2, score, class_id = placa_matricula
                    xcar1, ycar1, xcar2, ycar2, id_vehiculo = get_car(placa_matricula, ids_seguidos)

                    if id_vehiculo != -1:
                        placa_matricula_recortada = fotograma[int(y1):int(y2), int(x1): int(x2), :]
                        cv2.rectangle(fotograma, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                        placa_matricula_umbralizada = procesar_imagen_color(placa_matricula_recortada)
                        texto_placa_matricula, puntaje_texto_placa_matricula = leer_placa(placa_matricula_umbralizada)
                
                        texto_final=mejor_texto_reconocido(id_vehiculo,puntaje_texto_placa_matricula,texto_placa_matricula,mejor_puntaje, mejor_reconocimiento)
                    
                        cv2.putText(fotograma, texto_final, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                    
                        if texto_placa_matricula is not None:
                            resultados[numero_fotograma][id_vehiculo] = {
                                'vehiculo': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                'placa_matricula': {'bbox': [x1, y1, x2, y2],
                                                    'texto': texto_placa_matricula,
                                                    'bbox_puntaje': score,
                                                    'texto_puntaje': puntaje_texto_placa_matricula}
                            }

            
                
            # Escribir los resultados
            write_csv(resultados, './test2.csv')
            # Mostrar el fotograma con los rectángulos dibujados
            cv2.imshow('Vehículos Detectados', fotograma)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Liberar el objeto de captura y cerrar la ventana
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Configuración de la cámara PTZ IP de Dahua
    dahua_ip = '192.168.1.108'  # Cambia la dirección IP por la de tu cámara
    dahua_username = 'admin'
    dahua_password = 'Univida123+'

    # Mostrar el video en vivo
    show_dahua_live_video(dahua_ip, dahua_username, dahua_password)

if __name__ == "__main__":
    main()
