from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
import os
import sys
import argparse
import cv2
from PIL import Image
import torch
from torch.autograd import Variable
import math

def Convertir_RGB(img):       # Transformacion BGR a RGB
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img

def Convertir_BGR(img):         # Transformacion RGB a BGR
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #activar cuda
model = Darknet("config/tesis.cfg", img_size=416).to(device) #importar archivos de entrenamiento
model.load_darknet_weights("weights/yolov3_tesis.weights")
model.eval()  
clases = load_classes("data/clases.names")
Tensor_left = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

resultados=open("resultados.txt","w")     #retroalimentacion al usuario
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  #entrada
#cap =cv2.VideoCapture('feriatesis.mp4')  #descomentar para entrada de video
out = cv2.VideoWriter('borradortesis1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, (640,480))    
while cap:
    ret, frame = cap.read()
        
    if ret is False:
        break
        
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
    start_time = time.time() #inicio fps
    cv2.rectangle(frame,(0,40),(frame.shape[1],0),(136,130,130),-1)
    Alerta = "No se han detectado objetos " # alerta sin deteccion
    colorMensaje = (0,0,255)
    RGBimg=Convertir_RGB(frame) #conversion BGR a RGB
    sector_restringido = np.array([[100,40], [540,40], [540,340], [100,340]]) #Sector definido
    area_sector_restringido= (540-100)*(340-40)
    imagen_auxiliar = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imagen_auxiliar = cv2.drawContours(imagen_auxiliar, [sector_restringido], -1, (255), -1)
    image_area = cv2.bitwise_and(RGBimg,RGBimg, mask=imagen_auxiliar) #imagen restringida
    imgTensor_left = transforms.ToTensor()(image_area)
    imgTensor_left, _ = pad_to_square(imgTensor_left, 0)
    imgTensor_left = resize(imgTensor_left, 416)
    imgTensor_left = imgTensor_left.unsqueeze(0)
    imgTensor_left = Variable(imgTensor_left.type(Tensor_left))

    with torch.no_grad():
        detecciones = model(imgTensor_left)
        detecciones = non_max_suppression(detecciones, 0.8, 0.4)

    for deteccion in detecciones:
        if deteccion is not None:
            deteccion = rescale_boxes(deteccion, 416, RGBimg.shape[:2])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in deteccion:
                box_w = x2 - x1
                box_h = y2 - y1
                colorMensaje= (0,255,0)
                frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), (0,255,0), 1)
                Alerta = "Objeto Detectado"  #nuevo mensaje
                cv2.rectangle(frame,(x1, y1-25),((x1+(len(clases[int(cls_pred)]))*15),y2-box_h), 
                (0,255,0), -1)
                cv2.rectangle(frame,(x1, y1), ((x1+(len(clases[int(cls_pred)]))*15),y1 +30), 
                (0,255,0), -1) 
                cv2.putText(frame, clases[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,0.8, 
                (0,0,0), 1)# Nombre de la clase detectada
                cv2.putText(frame, str("%.2f" % float(conf*100))+"%", (x1, y1+30), 
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0), 1) # Certeza de prediccion de la clase
                    
                area_bbox= (x2-x1)*(y2-y1)
                area_porcentual= (area_bbox/area_sector_restringido)*100
                area_porcentual=area_porcentual.item()
                          
                if area_porcentual > 50: #evita solapamiento
                       
                    Mensaje = "Alerta!!! " + clases[int(cls_pred)]+ " MUY CERCA" # nuevo mensaje
                    frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), (255,0,0), 1)
                    cv2.rectangle(frame, (x1, y1-25), ((x1+(len(clases[int(cls_pred)]))*15) 
                    , y2-box_h), (255,0,0), -1)
                    cv2.rectangle(frame, (x1, y1), (((x1+(len(clases[int(cls_pred)]))*15)), 
                    y1 +30), (255,0,0), -1)
                    cv2.putText(frame, clases[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,0,0), 1)
                    cv2.putText(frame, str("%.2f" % float(conf*100))+"%", (x1, y1+30),
                    cv2.FONT_HERSHEY_SIMPLEX,  0.7,(0,0,0), 1)
                    cv2.putText(frame, Mensaje, (130, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0) ,2)
                    resultados.write("Se detectó " + clases[int(cls_pred)]+ " MUY CERCA\n")  #feedback   
             
    cv2.putText(frame, Alerta, (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colorMensaje ,2)
    cv2.drawContours(frame, [sector_restringido], -1, (0,0,255), 2)
    fps = 1.0 / (time.time() - start_time)      
    print("FPS: %.2f" % fps)   
    cv2.imshow('Resultado', Convertir_BGR(RGBimg))
    out.write(frame)
            
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
out.release()
cap.release()
cv2.destroyAllWindows()
resultados.close()
