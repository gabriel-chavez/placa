import string
import easyocr
import cv2
import numpy as np


# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(resultados, ruta_salida):
    """
    Escribe los resultados en un archivo CSV.

    Args:
        results (dict): Diccionario que contiene los resultados.
        output_path (str): Ruta al archivo CSV de salida.
    """
    with open(ruta_salida, 'w') as archivo_csv:
        archivo_csv.write('{},{},{},{},{},{},{}\n'.format('numero_fotograma', 'id_vehiculo', 'bbox_vehiculo',
                                                        'bbox_placa_matricula', 'puntaje_bbox_placa_matricula', 'numero_matricula',
                                                        'puntaje_numero_matricula'))

        for numero_fotograma in resultados.keys():
            for id_vehiculo in resultados[numero_fotograma].keys():
                #print(resultados[numero_fotograma][id_vehiculo])
                if 'vehiculo' in resultados[numero_fotograma][id_vehiculo].keys() and \
                'placa_matricula' in resultados[numero_fotograma][id_vehiculo].keys() and \
                'texto' in resultados[numero_fotograma][id_vehiculo]['placa_matricula'].keys():
                    archivo_csv.write('{},{},{},{},{},{},{}\n'.format(numero_fotograma,
                                                                        id_vehiculo,
                                                                        '[{} {} {} {}]'.format(
                                                                            resultados[numero_fotograma][id_vehiculo]['vehiculo']['bbox'][0],
                                                                            resultados[numero_fotograma][id_vehiculo]['vehiculo']['bbox'][1],
                                                                            resultados[numero_fotograma][id_vehiculo]['vehiculo']['bbox'][2],
                                                                            resultados[numero_fotograma][id_vehiculo]['vehiculo']['bbox'][3]),
                                                                        '[{} {} {} {}]'.format(
                                                                            resultados[numero_fotograma][id_vehiculo]['placa_matricula']['bbox'][0],
                                                                            resultados[numero_fotograma][id_vehiculo]['placa_matricula']['bbox'][1],
                                                                            resultados[numero_fotograma][id_vehiculo]['placa_matricula']['bbox'][2],
                                                                            resultados[numero_fotograma][id_vehiculo]['placa_matricula']['bbox'][3]),
                                                                        resultados[numero_fotograma][id_vehiculo]['placa_matricula']['bbox_puntaje'],
                                                                        resultados[numero_fotograma][id_vehiculo]['placa_matricula']['texto'],
                                                                        resultados[numero_fotograma][id_vehiculo]['placa_matricula']['texto_puntaje'])
                                    )
    archivo_csv.close()


# def license_complies_format(text):
#     """
#     Comprueba si el texto de la matrícula cumple con el formato requerido.

#     Args:
#         text (str): Texto de la matrícula.

#     Returns:
#         bool: True si la matrícula cumple con el formato, False en caso contrario.
#     """

   
#     if len(text) != 7:
#         return False

#     if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
#        (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
#        (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
#        (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
#        (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
#        (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
#        (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
#         return True
#     else:
#         return False
def license_complies_format(text):
    """
    Comprueba si el texto de la matrícula cumple con el formato requerido.

    Args:
        text (str): Texto de la matrícula.

    Returns:
        bool: True si la matrícula cumple con el formato, False en caso contrario.
    """
    if len(text)==6:
        if (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0] in dict_char_to_int.keys()) and \
        (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
        (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
        (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
        (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
        (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()):
            return True
        else:
            return False
        
    if len(text)==7:
        if (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0] in dict_char_to_int.keys()) and \
        (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
        (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
        (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
        (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
        (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
        (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
            return True
        else:
            return False
   
    return False


def format_license(text):
    """
    Formatea el texto de la matrícula convirtiendo caracteres utilizando los diccionarios de mapeo.

    Args:
        text (str): Texto de la matrícula.

    Returns:
        str: Texto de la matrícula formateado.
    """
    license_plate_ = ''
    if len(text)==6:
        mapping = {0: dict_char_to_int, 1: dict_char_to_int, 2: dict_char_to_int, 3: dict_int_to_char, 4: dict_int_to_char,
                    5: dict_int_to_char}
        for j in [0, 1, 2, 3, 4, 5]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]        

    if len(text)==7:
        mapping = {0: dict_char_to_int, 1: dict_char_to_int, 2: dict_char_to_int, 3: dict_char_to_int, 4: dict_int_to_char,
                    5: dict_int_to_char, 6: dict_int_to_char}
        for j in [0, 1, 2, 3, 4, 5, 6]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
        
    return license_plate_

def read_license_plate(license_plate_crop):
    """
    Lee el texto de la matrícula a partir de la imagen recortada proporcionada.

    Args:
        license_plate_crop (PIL.Image.Image): Imagen recortada que contiene la matrícula.

    Returns:
        tuple: Tupla que contiene el texto de la matrícula formateado y su puntaje de confianza.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score
    return None, None
 


def get_car(license_plate, vehicle_track_ids):
    """
    Recupera las coordenadas del vehículo y su ID basándose en las coordenadas de la matrícula.

    Args:
        license_plate (tuple): Tupla que contiene las coordenadas de la matrícula (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): Lista de IDs de seguimiento de vehículos y sus coordenadas correspondientes.

    Returns:
        tuple: Tupla que contiene las coordenadas del vehículo (x1, y1, x2, y2) y su ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1

def procesar_imagen(license_plate_crop):
    
    # Escalado de la imagen
    alto, ancho = license_plate_crop.shape[:2]
    factor_escala = 300 / max(alto, ancho)

    license_plate_crop = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    imagen_redimensionada = cv2.resize(license_plate_crop, None, fx=factor_escala, fy=factor_escala)

    # Aplica un filtro Gaussiano para suavizar la imagen
    #imagen_suavizada = cv2.GaussianBlur(imagen_redimensionada, (5, 5), 0)

    # Aplica un kernel de alta frecuencia
    kernel = np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]])
    imagen_alta_frecuencia = cv2.filter2D(imagen_redimensionada, -1, kernel)
    
    return imagen_alta_frecuencia