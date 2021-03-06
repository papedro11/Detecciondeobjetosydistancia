import os
import cv2
import numpy as np
import time
import darknet

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2))) #transformar flotantes
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

netMain = darknet.load_net_custom("./cfg/tesis_yolov4.cfg".encode("ascii"), #importar archivos de 
 "./yolov4_tesis.weights".encode("ascii"), 0, 1)                            # de entrenamiento
metaMain = darknet.load_meta("./cfg/tesis_yolov4.data".encode("ascii"))
network, class_names, class_colors = darknet.load_network("./cfg/tesis_yolov4.cfg", 
 "./cfg/tesis_yolov4.data", "./yolov4_tesis.weights", batch_size=1)

resultados=open("resultados.txt","w") #retroalimentacion al usuario
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)     #entrada
#cap = cv2.VideoCapture("feriatesis.mp4")     #descomentar para entrada de videoo
out = cv2.VideoWriter('borradortesis1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, (640,480))
    
darknet_image = darknet.make_image(640 , 480, 3) 
while True:                                                      
    prev_time = time.time()
    ret, frame = cap.read()                                 
       
    if not ret:                                                  
        break
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #conversion de BGR a RGB
    frame_resized = cv2.resize(frame_rgb,(640, 480), interpolation=cv2.INTER_CUBIC)
    darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())                
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.8) 
    
    for label, confidence, bbox in detections:
       x, y, w, h = (bbox[0],bbox[1],bbox[2],bbox[3])
       name_tag = label
       xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
       pt1 = (xmin, ymin)
       pt2 = (xmax, ymax)
       cv2.rectangle(frame_resized, pt1, pt2, (0,255,0), 1)
       cv2.putText(frame_resized, name_tag  , (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
       0.5,(0,255,0), 2)# Nombre de la clase detectada      
       cv2.putText(frame_resized, str("%.2f" % float(confidence))+"%",(pt1[0], pt1[1] + 13), 
       cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2) # Prediccion de la clase
       resultados.write("Se detect?? {} en X1: {}, Y1: {}, X2: {}, Y2: {}\n".format((name_tag),
       xmin, ymin, xmax, ymax )) 

    frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    print(1/(time.time()-prev_time))
    cv2.imshow('Demo', frame_resized) 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cv2.waitKey(3)
    out.write(frame_resized) 
out.release()
resultados.close()

