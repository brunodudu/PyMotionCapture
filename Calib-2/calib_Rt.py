import numpy as np
import cv2
import json
import argparse
import os

def ponto_medio_retas(reta1, reta2):
    # Vetor entre as origens
    w0 = reta2[0] - reta1[0]
    
    # Produto escalar entre os vetores diretores
    a = np.dot(reta1[1], reta1[1])  # ||v1||^2
    b = np.dot(reta1[1], reta2[1])  # v1 . v2
    c = np.dot(reta2[1], reta2[1])  # ||v2||^2
    d = np.dot(reta1[1], w0)  # v1 . w0
    e = np.dot(reta2[1], w0)  # v2 . w0
    
    # Determinante
    denom = a * c - b * b
    
    # Evitar divisão por zero (caso as retas sejam paralelas)
    if np.isclose(denom, 0):
        raise ValueError("As retas são paralelas e não possuem ponto único de menor distância.")
    
    # Parâmetros t e s que minimizam a distância entre as retas
    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom
    
    # Pontos mais próximos em cada reta
    ponto_r1 = reta1[0] + t * reta1[1]
    ponto_r2 = reta2[0] + s * reta2[1]
    
    # Ponto médio
    ponto_medio = (ponto_r1 + ponto_r2) / 2
    
    return ponto_medio

def reta3(K_inv, R_t, t, pixel):
    pixel_RP2 = [pixel[0], pixel[1], 1 ]
    p0 = - R_t @ np.transpose(t)
    pv = R_t @ K_inv @ np.transpose(pixel_RP2)
    return (p0, pv)

# Configuração do argparse para receber parâmetros da linha de comando
parser = argparse.ArgumentParser(description="Encontrar a matriz rotacao e a posicao relativa da fonte secundaria de video.")
parser.add_argument("principal_source", type=str, help="Fonte principal do vídeo.")
parser.add_argument("secundary_source", type=str, help="Fonte secundaria do vídeo.")
parser.add_argument("camera_distance", type=str, help="Distancia entre as cameras.")
args = parser.parse_args()

# Determinar se a fonte é um índice de webcam ou um arquivo de vídeo
try:
    source_0 = int(args.principal_source)  # Tenta converter para inteiro
except ValueError:
    source_0 = args.principal_source  # Se falhar, mantém como string (caminho do vídeo)

# Determinar se a fonte é um índice de webcam ou um arquivo de vídeo
try:
    source_1 = int(args.secundary_source)  # Tenta converter para inteiro
except ValueError:
    source_1 = args.secundary_source  # Se falhar, mantém como string (caminho do vídeo)

try:
    camera_distance = float(args.camera_distance)
except ValueError:
    camera_distance = args.camera_distance

with open("results/shots_for_FundamentalMatrix.json", "r") as json_file:
    pre_results = np.array(json.load(json_file)) 

with open(f"../Calib-1/results/camera_matrix_{source_0}.json", "r") as json_file:
    K_0 = np.array(json.load(json_file))

with open(f"../Calib-1/results/camera_matrix_{source_1}.json", "r") as json_file:
    K_1 = np.array(json.load(json_file))


K_0_inv = np.linalg.inv(K_0)
K_1_T = np.transpose(K_1)
K_1_inv = np.linalg.inv(K_1)

points_camera_0 = []
points_camera_1 = []

for result in pre_results:
    if f'Camera_{source_0}' in result and f'Camera_{source_1}' in result:
        points_camera_0.append(result[f'Camera_{source_0}'])
        points_camera_1.append(result[f'Camera_{source_1}'])

points_camera_0 = np.array(points_camera_0, dtype=np.float32)
points_camera_1 = np.array(points_camera_1, dtype=np.float32)
 
F, mask = cv2.findFundamentalMat(points_camera_0, points_camera_1, method=cv2.FM_RANSAC)

Tx_R = K_1_T @ F @ K_0

U, D, Vt = np.linalg.svd(Tx_R, full_matrices=True, compute_uv=True)

Z = [[0, 1, 0],
     [-1, 0, 0],
     [0, 0, 0]]
W = [[0, -1, 0],
     [1, 0, 0],
     [0, 0, 1]]

Tx = U @ Z @ np.transpose(U)
R1 = U @ W @ Vt
R2 = U @ np.transpose(W) @ Vt

t1 = U @ np.transpose([0, 0, 1])
t2 = -t1

R_f = None
t_f = None
Y_0 = reta3(K_0_inv, np.eye(3), np.zeros(3), points_camera_0[0])

for R in (R1, R2):
    for t in (t1, t2):
        Y = reta3(K_1_inv, np.transpose(R), t, points_camera_1[0])
        rec = ponto_medio_retas(Y_0, Y)
        rec_sec = np.transpose(R) @ (rec - t)
        if rec[2] > 0 and rec_sec[2] > 0:
            R_f = R
            t_f = t

print(R_f)
os.makedirs("results", exist_ok=True)
with open(f"results/RotationMatrix.json", "w") as json_file:
    json.dump(R_f.tolist(), json_file)

norm = np.linalg.norm(t_f)
t_f = (camera_distance / norm) * t_f
print(t_f)
with open(f"results/PositionVector.json", "w") as json_file:
    json.dump(t_f.tolist(), json_file)