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
        archivo_csv.write('{};{};{};{};{};{};{}\n'.format('numero_fotograma', 'id_vehiculo', 'bbox_vehiculo',
                                                        'bbox_placa_matricula', 'puntaje_bbox_placa_matricula', 'numero_matricula',
                                                        'puntaje_numero_matricula'))

        for numero_fotograma in resultados.keys():
            for id_vehiculo in resultados[numero_fotograma].keys():
                #print(resultados[numero_fotograma][id_vehiculo])
                if 'vehiculo' in resultados[numero_fotograma][id_vehiculo].keys() and \
                'placa_matricula' in resultados[numero_fotograma][id_vehiculo].keys() and \
                'texto' in resultados[numero_fotograma][id_vehiculo]['placa_matricula'].keys():
                    archivo_csv.write('{};{};{};{};{};{};{}\n'.format(numero_fotograma,
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

def leer_placa(license_plate_crop):
    """
    Lee el texto de la matrícula a partir de la imagen recortada proporcionada.

    Args:
        license_plate_crop (PIL.Image.Image): Imagen recortada que contiene la matrícula.

    Returns:
        tuple: Tupla que contiene el texto de la matrícula formateado y su puntaje de confianza.
    """
   
     # Calcular el promedio de confianza y concatenar todos los textos
    resultados_ocr = reader.readtext(license_plate_crop, allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    # Encontrar primer componente que cumple con el formato
    componentes_validos = [componente for componente in resultados_ocr if license_complies_format(componente[1])]

    # Si no hay componentes válidos, concatenar textos y calcular promedio de confianza
    if not componentes_validos:
        textos = [componente[1] for componente in resultados_ocr]
        texto_concatenado = ''.join(textos)

        # Validar formato del texto concatenado
        if license_complies_format(texto_concatenado):
            promedio_confianza = sum(componente[2] for componente in resultados_ocr) / len(resultados_ocr)
            return format_license(texto_concatenado), round(promedio_confianza*100,2)
        else:
            return '-----', 0
    else:
        # Retornar solo el primer componente válido
        componente_valido = componentes_validos[0]
        return format_license(componente_valido[1]), round(componente_valido[2]*100,2)


 


def get_car(license_plate, vehicle_track_ids):
    """
    Recupera las coordenadas del vehículo y su ID basándose en las coordenadas de la matrícula.

    Args:
        license_plate (tuple): Tupla que contiene las coordenadas de la matrícula (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): Lista de IDs de seguimiento de vehículos y sus coordenadas correspondientes.

    Returns:
        tuple: Tupla que contiene las coordenadas del vehículo (x1, y1, x2, y2) y su ID.
    """
    if (vehicle_track_ids == 0).any():
        return -1, -1, -1, -1, -1
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

def procesar_imagen_2(placa_recortada):
    
    
    # Aplicar ajuste de contraste y brillo
    imagen_ajustada = cv2.convertScaleAbs(placa_recortada, alpha=1.5, beta=50)
    #imagen_ajustada = cv2.convertScaleAbs(imagen_original, alpha=1, beta=50)
    # Aplicar filtrado bilateral para reducir el ruido
    imagen_filtrada = cv2.bilateralFilter(imagen_ajustada, 11, 17, 17)

    # Convertir la imagen original a escala de grises
    imagen_gris = cv2.cvtColor(imagen_filtrada, cv2.COLOR_BGR2GRAY)

    # Aplicar ecualización adaptativa del histograma
    imagen_ecualizada = cv2.equalizeHist(imagen_gris)

    #_, imagen_ecualizada = cv2.threshold(imagen_ecualizada, 140, 255, cv2.THRESH_BINARY_INV)

    return imagen_ecualizada

def procesar_imagen_ajustes(placa_recortada):
          
    placa_recortada = cv2.cvtColor(placa_recortada, cv2.COLOR_BGR2GRAY)
    placa_recortada = cv2.threshold(placa_recortada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return placa_recortada

def procesar_imagen_color(placa_recortada):
          
    
    # Cargar imagen, convertir a HSV y aplicar umbral de color

    hsv = cv2.cvtColor(placa_recortada, cv2.COLOR_BGR2HSV)
    lower = np.array([1,30,60])
    upper= np.array([179,255,160])
    mask = cv2.inRange(hsv, lower, upper)

    # Invertir la imagen y realizar OCR
    invert = 255 - mask
    return invert
def mejor_texto_reconocido(id_vehiculo, puntaje_texto_placa_matricula, texto_placa_matricula,  mejor_puntaje, mejor_reconocimiento):
    id_vehiculo1 = int(id_vehiculo)
    if puntaje_texto_placa_matricula is None:
        puntaje_texto_placa_matricula = 0
    if puntaje_texto_placa_matricula > mejor_puntaje[id_vehiculo1]:
        mejor_reconocimiento[id_vehiculo1] = texto_placa_matricula
        mejor_puntaje[id_vehiculo1] = puntaje_texto_placa_matricula
        
    return f"{mejor_reconocimiento[id_vehiculo1]} ({mejor_puntaje[id_vehiculo1]}%)" if mejor_reconocimiento[id_vehiculo1] is not None else "---"
