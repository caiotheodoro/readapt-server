import os
import numpy as np
import cv2

AGE_MODEL = os.path.join(os.path.dirname(__file__), 'dataset/deploy_age.prototxt')  # AgeNet
AGE_PROTO = os.path.join(os.path.dirname(__file__), 'dataset/age_net.caffemodel')   #carrega o modelo
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)   #valores de entrada
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', 
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']   #intervalo de idade

FACE_PROTO =  os.path.join(os.path.dirname(__file__), "dataset/deploy.prototxt.txt")    #carrega o modelo
FACE_MODEL = os.path.join(os.path.dirname(__file__), "dataset/res10_300x300_ssd_iter_140000_fp16.caffemodel")   #carrega o modelo

frame_width = 1280  # largura da imagem
frame_height = 720  # altura da imagem

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL) #carrega o modelo
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)    #carrega o modelo


def get_faces(frame, confidence_threshold=0.5): # função que detecta faces
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))   #transforma a imagem em um blob
    face_net.setInput(blob)  #seta o blob como entrada
    output = np.squeeze(face_net.forward()) #executa o forward pass
    faces = []                         #cria uma lista de faces
    for i in range(output.shape[0]):    #percorre todas as linhas do output
        confidence = output[i, 2]       #pega a confiança da face
        if confidence > confidence_threshold:   #se a confiança for maior que o threshold
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])   #pega as coordenadas da face
            start_x, start_y, end_x, end_y = box.astype(int)     #converte as coordenadas para int
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10    #adiciona 10px aos limites da face
            start_x = 0 if start_x < 0 else start_x #se o início da face for menor que 0, seta o início da face como 0
            start_y = 0 if start_y < 0 else start_y #se o início da face for menor que 0, seta o início da face como 0
            end_x = 0 if end_x < 0 else end_x #se o fim da face for menor que 0, seta o fim da face como 0
            end_y = 0 if end_y < 0 else end_y #se o fim da face for menor que 0, seta o fim da face como 0 
            faces.append((start_x, start_y, end_x, end_y))  #adiciona a face à lista
    return faces                             #retorna a lista de faces

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):   #função que redimensiona a imagem
    dim = None  #dimensões da imagem
    (h, w) = image.shape[:2]    #pega as dimensões da imagem
    if width is None and height is None:    #se não for passado nenhum parâmetro
        return image        
    if width is None:          #se não for passado a largura
        r = height / float(h)   #calcula o fator de redimensionamento
        dim = (int(w * r), height)      #calcula as novas dimensões
    else:
        r = width / float(w)    #calcula o fator de redimensionamento
        dim = (width, int(h * r))   #calcula as novas dimensões
    return cv2.resize(image, dim, interpolation = inter)    #redimensiona a imagem


def predict_age(input_path: str):
    img = cv2.imread(input_path)    #carrega a imagem
    frame = img.copy()            #copia a imagem
    if frame.shape[1] > frame_width:    #se a largura da imagem for maior que a largura da tela
        frame = image_resize(frame, width=frame_width)  #redimensiona a imagem
    faces = get_faces(frame)    #pega as faces
    face = list()             #cria uma lista de faces
    face.append(faces[0])       #adiciona a primeira face à lista
    for i, (start_x, start_y, end_x, end_y) in enumerate(face):   
        face_img = frame[start_y: end_y, start_x: end_x]    #pega a face
        blob = cv2.dnn.blobFromImage(   
            image=face_img, scalefactor=1.0, size=(227, 227), 
            mean=MODEL_MEAN_VALUES, swapRB=False
        )   #transforma a imagem em um blob
        age_net.setInput(blob)  #seta o blob como entrada
        age_preds = age_net.forward()   #executa o forward pass
        i = age_preds[0].argmax()       #pega o índice do maior valor
        age = AGE_INTERVALS[i]        #pega o intervalo de idade
       
    return age  #retorna a idade
