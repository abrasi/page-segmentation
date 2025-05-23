import cv2
import numpy as np

def readImage(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    
    return image

def writeImage(filename, image):
    cv2.imwrite(filename, image)

# 8/16 CONTORNOS DE FOLIO DETECTADOS
def detect_paper(image):
    
    # Ecualización adaptativa
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    equalized = clahe.apply(image)
    
    # Filtrado bilateral
    bilateral = cv2.bilateralFilter(equalized, d=1, sigmaColor=75, sigmaSpace=75)

    # Suavizado Gaussiano
    blurr = cv2.GaussianBlur(bilateral, (5,5), 0)
    
    # Umbralización
    threshold = cv2.adaptiveThreshold(blurr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2)
    
    # Cierre morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    
    # Bordes con Canny
    edges = cv2.Canny(cleaned, 5, 90)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        
        # Se descubre el umbral de área para filtrar los contornos
        if cv2.contourArea(contour) > 5000000:
            
            # Se busca un trapezoide
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                print("Folio detectado")
                
                vertices = approx.reshape(4, 2)
                
                center = np.mean(vertices, axis=0)
                
                # Angulos respecto al centroide
                angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
                
                # Ordenar vértices según el ángulo
                sorted_vertices = vertices[np.argsort(angles)]
                    
                return (approx, sorted_vertices)
    cv2.drawContours(image, contours, -1, (255, 255, 255), 3)
    return False

def correct_perspective(image, vertices):
    
    src_points = np.float32(vertices)
    
    # Coordenadas de los vértices de destino
    width, height = 2500, 3500
    destination = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    
    # Matriz de transformación
    matrix = cv2.getPerspectiveTransform(src_points, destination)
    
    # Transformación de perspectiva
    warped = cv2.warpPerspective(image, matrix, (width, height))
    
    return warped

# Recortamos un poco la imagen para evitar bordes
def crop(image):
    
    top_margin = 50
    bottom_margin = 50
    left_margin = 50
    right_margin = 50

    height, width = image.shape[:2]

    cropped_image = image[top_margin:height-bottom_margin, left_margin:width-right_margin]
    
    return cropped_image

# Segmentado del texto
def text_segmentation(image):
    
    # Segmentación de colores para manchas
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Segmentación de rojo
    lower_red1 = np.array([0, 15, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 15, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Segmentación de azul
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])  
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask = cv2.bitwise_or(mask_red, mask_blue)

    # Todo lo que sea azul o rojo lo hacemos blanco
    image[mask > 0] = [255, 255, 255]
        
    # Refinado
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    blurr = cv2.GaussianBlur(gray, (9,9), 0)
    
    _, threshold = cv2.threshold(blurr, 110, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    
    # Reconstrucción del texto
    white_paper = np.ones_like(cleaned) * 255
    white_paper[cleaned == 255] = 0
    return white_paper