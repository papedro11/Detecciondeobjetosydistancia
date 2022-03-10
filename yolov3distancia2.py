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

DisB = 7   #distancia entre camaras           
f = 6      #distancia focal de camara       
alpha = 45  #angulo horizontal de camara     
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
cap_left = cv2.VideoCapture(1,cv2.CAP_DSHOW)
cap_right= cv2.VideoCapture(2,cv2.CAP_DSHOW)
#cap_right =cv2.VideoCapture('videotesisizquierda.mp4') #videos elaborados
#cap_left =cv2.VideoCapture('videotesisderecha.mp4')
out_left = cv2.VideoWriter('salidaizquierda.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, (416,416))
out_right= cv2.VideoWriter('salidaderecha.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, (416,416))
resultados=open("resultados_yolov3.txt","w")  
colores = np.random.randint(0, 255, size=(len(clases), 3), dtype="uint8") #colores randomizados
while True:
    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read() 
    #Calibrar una unica vez
    #frame_right, frame_left = calib.undistorted(frame_right, frame_left)         
    if ret_right==False or ret_left==False:                    
        break
    else:
        kernel = np.ones((5,5),np.float32)/25
        frame_left= cv2.filter2D(frame_left,-1,kernel) #quitar ruido de camaras
        frame_right= cv2.filter2D(frame_right,-1,kernel)  
        # deteccion en ambos frames
        frame_left = cv2.resize(frame_left, (416, 416), interpolation=cv2.INTER_CUBIC)
        RGBimg_left=Convertir_RGB(frame_left)
        imgTensor_left = transforms.ToTensor()(RGBimg_left)
        imgTensor_left, _ = pad_to_square(imgTensor_left, 0)
        imgTensor_left = resize(imgTensor_left, 416)
        imgTensor_left = imgTensor_left.unsqueeze(0)
        imgTensor_left = Variable(imgTensor_left.type(Tensor_left))
            
        frame_right = cv2.resize(frame_right, (416, 416), interpolation=cv2.INTER_CUBIC)
        RGBimg_right=Convertir_RGB(frame_right)
        imgTensor_right = transforms.ToTensor()(RGBimg_right)
        imgTensor_right, _ = pad_to_square(imgTensor_right, 0)
        imgTensor_right = resize(imgTensor_right, 416)
        imgTensor_right = imgTensor_right.unsqueeze(0)
        imgTensor_right = Variable(imgTensor_right.type(Tensor_left))
        start_time = time.time()
        with torch.no_grad():
            detections_left = model(imgTensor_left)
            detections_left = non_max_suppression(detections_left, 0.8, 0.4)
            detections_right = model(imgTensor_right)
            detections_right = non_max_suppression(detections_right, 0.8, 0.4)                        
        for detectionL in detections_left:             
          for detectionR in detections_right:
            if detectionL is not None and detectionR is not None:
               detectionL = rescale_boxes(detectionL, 416, RGBimg_left.shape[:2])
               detectionR = rescale_boxes(detectionR, 416, RGBimg_right.shape[:2])                   
               for x1L, y1L, x2L, y2L, confL, cls_confL, cls_predL in detectionL:
                  for x1R, y1R, x2R, y2R, confR, cls_confR, cls_predR in detectionR:
                      box_wL = x2L - x1L
                      box_hL = y2L - y1L
                      box_wR = x2R - x1R
                      box_hR = y2R - y1R
                      XmedioL = (x1L+x2L)/2
                      YmedioL=(y1L+y2L)/2
                      XmedioR=(x1R+x2R)/2
                      YmedioR=(y1R+y2R)/2
                      CentroL=(XmedioL,YmedioL)
                      CentroR=(XmedioR, YmedioR) 
                      colorL = [int(c) for c in colores[int(cls_predL)]]
                      #Calculo de disparidad
                      height_right, width_right, depth_right = RGBimg_right.shape 
                      height_left, width_left, depth_left = RGBimg_left.shape

                      if width_right == width_left:     #centimetro a pixel
                         f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180) 

                      else:
                           print('pixeles no coinciden')
                            
                      if cls_predL == cls_predR:
                         x_right = CentroR[0]
                         x_left = CentroL[0]
                         disparity = abs(x_left-x_right)
                         
                         if 25 < disparity < 250:
                         
                            formula = (DisB*f_pixel)/disparity
                            formula= formula.item()
                            Distancia=int(formula)
                                                           
                            cv2.putText(frame_left, "Distancia: " + str(round(Distancia,3))+" cm", 
                            (x1L, y1L+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)                               
                            cv2.putText(frame_right, "Distancia: " + str(round(Distancia,3))+" cm", 
                            (x1R, y1R+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
 
                            resultados.write("Se detectó " + clases[int(cls_predL)] +  " en " 
                            + str(round(Distancia,3)) + " cm\n")  
                         else:
                             print('mala disparidad')
                             
                      cv2.rectangle(frame_left, (x1L, y1L-25),((x1L+(len(clases[int(cls_predL)]))*17)
                      , y2L-box_hL), colorL, -1)
                      cv2.rectangle(frame_right, (x1R, y1R-25),((x1R+(len(clases[int(cls_predR)]))*17) 
                      , y2R-box_hR), colorL, -1)
                      frame_left = cv2.rectangle(frame_left, (x1L, y1L + box_hL),(x2L, y1L),colorL,2)
                      cv2.putText(frame_left, clases[int(cls_predL)], (x1L, y1L), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
                      frame_right = cv2.rectangle(frame_right,(x1R, y1R + box_hR),(x2R, y1R),colorL,2)
                      cv2.putText(frame_right, clases[int(cls_predR)], (x1R, y1R),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)         
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)            
        cv2.imshow('camara_izquierda', Convertir_BGR(RGBimg_left))
        cv2.imshow('camara_derecha', Convertir_BGR(RGBimg_right))
        out_left.write(RGBimg_left)
        out_right.write(RGBimg_right)       
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
out_left.release()
out_right.release()
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
resultados.close()
