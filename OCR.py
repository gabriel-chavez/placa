# Importamos la libreria OpenCv
import cv2

# Importamos Pytesseract
import pytesseract

# Abrimos la imagen
im = cv2.imread("BN\placa_94.0_258.png")

# Utilizamos el método "image_to_string"
# Le pasamos como argumento la imagen abierta con Pillow
texto = pytesseract.image_to_string(im)

# Mostramos el resultado
print(texto)