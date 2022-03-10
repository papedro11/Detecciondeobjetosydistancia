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

DisB = 7   #distancia entre camaras           
f = 6      #distancia focal de camara       
alpha = 45  #angulo horizontal de camara     

netMain = darknet.load_net_custom("./cfg/tesis_yolov4.cfg".encode("ascii"), #importar archivos de 
 "./yolov4_tesis.weights".encode("ascii"), 0, 1)                            # de entrenamiento
metaMain = darknet.load_meta("./cfg/tesis_yolov4.data".encode("ascii"))
network, class_names, class_colors = darknet.load_network("./cfg/tesis_yolov4.cfg", 
 "./cfg/tesis_yolov4.data", "./yolov4_tesis.weights", batch_size=1)
 
#cap_der = cv2.VideoCapture(2, cv2.CAP_DSHOW)  
#cap_izq = cv2.VideoCapture(1, cv2.CAP_DSHOW)        
                              
cap_der = cv2.VideoCapture("videotesisizq2.mp4")                            
cap_izq = cv2.VideoCapture("videotesisder2.mp4")  
resultados=open("resultadosdistancia2_yolov4.txt","w")    
out_der = cv2.VideoWriter('videotesis2derecha2.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, (416,416))
out_izq = cv2.VideoWriter('videotesis2izquierda2.mp4',cv2.VideoWriter_fourcc(*'mp4v')
, 20,(416,416))

darknet_image_izq = darknet.make_image(416 , 416, 3) 
darknet_image_der = darknet.make_image(416 , 416, 3) 

while True:                                                     
    prev_time = time.time()    
    ret_izq, frame_izq = cap_izq.read()    
    ret_der, frame_der = cap_der.read()                                
   
    if ret_der==False or ret_izq==False:                                               
        break       
    kernel = np.ones((5,5),np.float32)/25
    frame_izq= cv2.filter2D(frame_izq,-1,kernel)
    frame_der= cv2.filter2D(frame_der,-1,kernel)        
    frame_izq = cv2.resize(frame_izq, (416, 416), interpolation=cv2.INTER_CUBIC)
    frame_der = cv2.resize(frame_der, (416, 416), interpolation=cv2.INTER_CUBIC)
    frame_rgb_izq = cv2.cvtColor(frame_izq, cv2.COLOR_BGR2RGB)    
    frame_rgb_der = cv2.cvtColor(frame_der, cv2.COLOR_BGR2RGB)    
    darknet.copy_image_from_bytes(darknet_image_izq,frame_rgb_izq.tobytes())                
    darknet.copy_image_from_bytes(darknet_image_der,frame_rgb_der.tobytes())     
    detections_izq = darknet.detect_image(network, class_names, darknet_image_izq, thresh=0.8) 
    detections_der = darknet.detect_image(network, class_names, darknet_image_der, thresh=0.8)
    
    for label_izq, confidence_izq, bbox_izq in detections_izq:
       x_izq, y_izq, w_izq, h_izq = (bbox_izq[0],bbox_izq[1],bbox_izq[2],bbox_izq[3])
       name_tag_izq = label_izq
       xmin_izq, ymin_izq, xmax_izq, ymax_izq = convertBack(float(x_izq), float(y_izq), 
       float(w_izq), float(h_izq))
       pt1_izq = (xmin_izq, ymin_izq)
       pt2_izq = (xmax_izq, ymax_izq)
       Xmedio_izq=(xmin_izq + xmax_izq)/2
       Ymedio_izq=(ymin_izq + ymax_izq)/2
       Centro_izq=(Xmedio_izq , Ymedio_izq)     
       height_izq, width_izq, depth_izq = frame_rgb_izq.shape      
       cv2.rectangle(frame_rgb_izq, pt1_izq, pt2_izq, (0,255,0), 1)
       cv2.putText(frame_rgb_izq, name_tag_izq  , (pt1_izq[0], pt1_izq[1] - 5), 
       cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2) 
                 
       for label_der, confidence_der, bbox_der in detections_der:
          x_der, y_der, w_der, h_der = (bbox_der[0],bbox_der[1],bbox_der[2],bbox_der[3])
          name_tag_der = label_der
          xmin_der, ymin_der, xmax_der, ymax_der = convertBack(float(x_der), float(y_der), 
          float(w_der), float(h_der))
          pt1_der = (xmin_der, ymin_der)
          pt2_der = (xmax_der, ymax_der)
               
          Xmedio_der=(xmin_der + xmax_der)/2
          Ymedio_der=(ymin_der + ymax_der)/2
          Centro_der=(Xmedio_der , Ymedio_der)
               
          height_der, width_der, depth_der = frame_rgb_der.shape
               
          cv2.rectangle(frame_rgb_der, pt1_der, pt2_der, (0,255,0), 1)
          cv2.putText(frame_rgb_der, name_tag_der  , (pt1_der[0], pt1_der[1] - 5), 
          cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)     
          if width_der == width_izq:
             f_pixel = (width_der * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

          else:
              print('pixeles no coinciden')                                    
          if name_tag_der  == name_tag_izq:
             x_der = Centro_der[0]
             x_izq = Centro_izq[0]                          
             disparity = abs(x_izq-x_der)                    
             if 25 < disparity < 250:
                                  
                formula = (DisB*f_pixel)/disparity
                Distancia=int(formula)
                
                cv2.putText(frame_rgb_izq, "Distancia: " + str(round(Distancia,3))+" cm", 
                (xmin_izq, ymin_izq+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                cv2.putText(frame_rgb_der, "Distancia: " + str(round(Distancia,3))+" cm", 
                (xmin_der, ymin_der+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                resultados.write("Se detectó " + name_tag_izq +  " en " 
                + str(round(Distancia,3)) + " cm\n")                 
 
    frame_bgr_izq = cv2.cvtColor(frame_rgb_izq, cv2.COLOR_BGR2RGB)
    frame_bgr_der = cv2.cvtColor(frame_rgb_der, cv2.COLOR_BGR2RGB)
    print(1/(time.time()-prev_time))
    cv2.imshow('Cam_izq', frame_bgr_izq) 
    cv2.imshow('Cam_der', frame_bgr_der)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    cv2.waitKey(3)
    out_izq.write(frame_bgr_izq) 
    out_der.write(frame_bgr_der) 
cap_izq.release()    
cap_der.release()                                                 
out_izq.release()
out_der.release()
resultados.close()

