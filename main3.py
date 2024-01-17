from ultralytics import YOLO
import cv2
import numpy as np

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()
vehicles = [2, 3, 5, 7]
# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# load video
#cap = cv2.VideoCapture('./20231219_164345.mp4')
cap = cv2.VideoCapture('./VID_2.mp4')

# Define las nuevas dimensiones (ancho, alto) para la ventana redimensionada
nuevas_dimensiones_ventana = (540, 380)  # Puedes ajustar estas dimensiones según tus necesidades

# Crea la ventana de salida con el nombre 'Detected Vehicles'
cv2.namedWindow('Detected Vehicles', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detected Vehicles', nuevas_dimensiones_ventana[0], nuevas_dimensiones_ventana[1])

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # # Redimensionar el fotograma con especificación de método de interpolación
        #frame = cv2.resize(frame, nuevas_dimensiones_ventana, interpolation=cv2.INTER_NEAREST)

        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

                # Dibujar un recuadro alrededor del vehículo con un color blanco y grosor 3
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            if car_id != -1:

                
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                
                


                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 200), 2)
                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                # Agregar el texto sobre el cuadro de la placa
                if license_plate_text is not None:
                    cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    # Guardar la imagen de la placa
                    filename = f'placa_{car_id}_{frame_nmr}.png'
                    cv2.imwrite(filename, license_plate_crop_gray)
                else :
                    cv2.putText(frame, "NN", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

        # write results
        write_csv(results, './test2.csv')

        # Mostrar el fotograma con los recuadros dibujados
        cv2.imshow('Detected Vehicles', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
