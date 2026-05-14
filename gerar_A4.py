import cv2
import numpy as np
import os

A4_WIDTH = 2480
A4_HEIGHT = 3508
DPI = 300
CM_TO_PX = DPI / 2.54 

MARKER_SIZE_PX = int(5.0 * CM_TO_PX) 
MARGIN_PX = int(2.0 * CM_TO_PX)       

folha = np.ones((A4_HEIGHT, A4_WIDTH), dtype=np.uint8) * 255

FOLDER_NAME = "images"

arquivos = [f"aruco-marker-ID={i}.png" for i in range(6)]

x_offset = MARGIN_PX
y_offset = MARGIN_PX

print(f"Buscando marcadores na pasta: {os.path.abspath(FOLDER_NAME)}")

for i, nome_arq in enumerate(arquivos):
    caminho_completo = os.path.join(FOLDER_NAME, nome_arq)
    
    if not os.path.exists(caminho_completo):
        print(f"Erro: {nome_arq} não encontrado em {FOLDER_NAME}")
        continue
    
    # 
    img_marker = cv2.imread(caminho_completo, cv2.IMREAD_GRAYSCALE)
    img_res = cv2.resize(img_marker, (MARKER_SIZE_PX, MARKER_SIZE_PX))
    
    # Insere na folha A4
    folha[y_offset:y_offset+MARKER_SIZE_PX, x_offset:x_offset+MARKER_SIZE_PX] = img_res
    
    # 2 colunas
    if (i + 1) % 2 == 0:
        x_offset = MARGIN_PX
        y_offset += MARKER_SIZE_PX + MARGIN_PX
    else:
        x_offset += MARKER_SIZE_PX + MARGIN_PX


output_name = "folha_impressao_final.png"
cv2.imwrite(output_name, folha)

print(f"Folha gerada: {output_name}")