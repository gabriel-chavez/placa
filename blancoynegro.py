import cv2
import numpy as np

# Lee la imagen
imagen_original = cv2.imread('ruta_de_la_imagen.jpg', cv2.IMREAD_GRAYSCALE)  # Reemplaza con la ruta de tu imagen

# Aplica una transformación de umbral (binarización) para mejorar el contraste
_, imagen_umbral = cv2.threshold(imagen_original, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Aplica un filtro Gaussiano para suavizar la imagen y reducir el ruido
imagen_suavizada = cv2.GaussianBlur(imagen_umbral, (5, 5), 0)

# Aplica la detección de bordes (opcional)
imagen_bordes = cv2.Canny(imagen_suavizada, 50, 150)

# Encuentra y dibuja los contornos de la imagen
contornos, _ = cv2.findContours(imagen_bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imagen_original, contornos, -1, (0, 255, 0), 2)

# Muestra las imágenes
cv2.imshow('Imagen Original', imagen_original)
cv2.imshow('Imagen Binarizada', imagen_umbral)
cv2.imshow('Imagen Suavizada', imagen_suavizada)
cv2.imshow('Imagen de Bordes', imagen_bordes)

cv2.waitKey(0)
cv2.destroyAllWindows()
