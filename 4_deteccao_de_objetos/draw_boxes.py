import numpy as np
import cv2


COLORS = [(255,0,0),
         (0,255,0),
         (0,0,255),
         (255,255,0),
         (255,0,255),
         (0,255,255),
         (255,255,255),
         (125,255,125),
         (255,200,40),
         (255,40,200),
         (40,200,255)]


def boxing(original_img, predictions):
    newImage = np.copy(original_img)

    for index, result in enumerate(predictions):
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

       
        newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), COLORS[index], 2)
        newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (255, 255, 255), 2, cv2.LINE_8)
            
    return newImage