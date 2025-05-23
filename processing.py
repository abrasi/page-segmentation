import utils
import numpy as np
import cv2


# PRUEBA DE TODAS LAS IMAGENES
for i in range(1, 17):
    image = cv2.imread('./materiales/doc' + str(i) + '.jpg')
    copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    paper = utils.detect_paper(gray)
    
    if paper:
        
        contour, vertices = paper
        cv2.drawContours(copy, [contour], -1, (0, 255, 0), 5)
        
        for vertex in vertices:
            cv2.circle(copy, tuple(vertex), 20, (0, 0, 255), -1)
            
        compare = np.hstack((image, copy))
        
        utils.writeImage('./materiales/rebuilt/doc' + str(i) + '_paper_detected.jpg', compare)   
        
        perspective = utils.correct_perspective(image, vertices) 
        
        # Recortamos un poco la imagen
        cropped = utils.crop(perspective)
        
        text = utils.text_segmentation(cropped)
        
        utils.writeImage('./materiales/rebuilt/doc' + str(i) + '_text_segmented.jpg', text)
    else:
        utils.writeImage('./materiales/rebuilt/doc' + str(i) + '_paper_not_detected.jpg', gray)



# PRUEBA PARA 1 IMAGEN
# i = 5
# image = cv2.imread('./materiales/doc' + str(i) + '.jpg')
# copy = image.copy()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# paper = utils.detect_paper(gray)
# if paper:
#     contour, vertices = paper
#     cv2.drawContours(copy, [contour], -1, (0, 255, 0), 5)
    
#     for vertex in vertices:
#         cv2.circle(copy, tuple(vertex), 20, (0, 0, 255), -1)
        
#     compare = np.hstack((image, copy))
    
#     utils.writeImage('./materiales/rebuilt/doc' + str(i) + '_paper_detected.jpg', compare)   
    
#     perspective = utils.correct_perspective(image, vertices) 
    
#     # Recortamos un poco la imagen
#     cropped = utils.crop(perspective)
    
#     text = utils.text_segmentation(cropped)
    
#     utils.writeImage('./materiales/rebuilt/doc' + str(i) + '_text_segmented.jpg', text)
# else:
#     utils.writeImage('./materiales/rebuilt/doc' + str(i) + '_paper_not_detected.jpg', gray)
