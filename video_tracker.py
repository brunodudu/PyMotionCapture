import cv2
import threading
from inference_sdk import InferenceHTTPClient
import numpy as np
import argparse
import json

project_id = "balls-obj-det-3.0-euven"
model_version = 2
api_key = "S3lt1G4Wx3nBEACDxu9z"
api_url = "http://localhost:9001"

client = InferenceHTTPClient(api_url=api_url, api_key=api_key)

# Configuração do argparse para receber parâmetros da linha de comando
parser = argparse.ArgumentParser(description="Encontrar posicao do objeto.")
parser.add_argument("principal_source", type=str, help="Fonte principal do vídeo.")
parser.add_argument("secundary_source", type=str, help="Fonte secundaria do vídeo.")
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

with open(f"Calib-1/results/camera_matrix_{source_0}.json", "r") as json_file:
    K_0 = np.array(json.load(json_file))

with open(f"Calib-1/results/camera_matrix_{source_1}.json", "r") as json_file:
    K_1 = np.array(json.load(json_file))

R_0 = np.eye(3)
t_0 = np.zeros(3)

with open(f"Calib-2/results/RotationMatrix.json", "r") as json_file:
    R_1 = np.array(json.load(json_file))

with open(f"Calib-2/results/PositionVector.json", "r") as json_file:
    t_1 = np.array(json.load(json_file))

K_0_inv = np.linalg.inv(K_0)
K_1_inv = np.linalg.inv(K_1)

def ponto_medio_retas(reta1, reta2):
    """
    Calcula o ponto médio entre os pontos mais próximos de duas retas no espaço 3D.
    
    Parâmetros:
    - p0_1: np.array, ponto de origem da reta 1
    - v1: np.array, vetor diretor da reta 1
    - p0_2: np.array, ponto de origem da reta 2
    - v2: np.array, vetor diretor da reta 2
    
    Retorna:
    - ponto_medio: np.array, ponto médio entre os pontos mais próximos das retas
    """
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

def reta3D(K_inv, R_t, t, pixel):
    pixel_RP2 = [pixel[0], pixel[1], 1 ]
    p0 = - R_t @ np.transpose(t)
    pv = R_t @ K_inv @ np.transpose(pixel_RP2)
    return (p0, pv)


def desenhar_centro(image, center_x, center_y, cor):
    line_length = 10
    
    # Desenhar a linha horizontal do '+'
    cv2.line(image, (int(center_x - line_length // 2), center_y), (int(center_x + line_length // 2), center_y),  cor, 2)  # Verde

    # Desenhar a linha vertical do '+'
    cv2.line(image, (center_x, int(center_y - line_length // 2)), (center_x, int(center_y + line_length // 2)),  cor, 2)


def video_thread(source, string_display, stop_event):
    global client
    global project_id
    global model_version
    global result_threads
    global results_shots

    cap = cv2.VideoCapture(source)
    while not stop_event.is_set():
        ret, image = cap.read()
        if not ret:
            break

        results = client.infer(image, model_id=f"{project_id}/{model_version}")

        if len(results['predictions']) == 0:
            result_threads[string_display] = (0, 0)

        for prediction in results['predictions']:
                        
            width, height = prediction['width'], prediction['height']
            x, y = int(prediction['x'] - width/2) , int(prediction['y'] - height/2)
            
            class_name = prediction['class']

            # Calculate the bottom right x and y coordinates
            x2 = int(x + width)
            y2 = int(y + height)

            if class_name == 'RedBall':
                cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 3)
                desenhar_centro(image, int(prediction['x']), int(prediction['y']), (0, 0, 255))
                result_threads[string_display] = (prediction['x'], prediction['y'])

        cv2.imshow(string_display, image)

        # Verifica se a tecla 's' foi pressionada para salvar os resultados
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            stop_event.set()
            break

    
    cap.release()
    cv2.destroyWindow(string_display)

def position_thread(stop_event):
    global result_threads
    while not stop_event.is_set():
        pixel_0 = result_threads[f"Camera_{source_0}-principal"]
        pixel_1 = result_threads[f"Camera_{source_1}-secundaria"]
        if pixel_0 != (0, 0) and pixel_1 != (0, 0):
            reta0 = reta3D(K_0_inv, np.transpose(R_0), t_0, pixel_0)
            reta1 = reta3D(K_1_inv, np.transpose(R_1), t_1, pixel_1)
            rec = ponto_medio_retas(reta0, reta1)
            print(100*rec)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            stop_event.set()
            break

# Evento para parar as threads
stop_event = threading.Event()

result_threads = {
    f"Camera_{source_0}-principal": (0,0),
    f"Camera_{source_1}-secundaria": (0,0)
}
results_shots = []

# Iniciando as threads para as câmeras
thread_0 = threading.Thread(target=video_thread, args=(source_0, f"Camera_{source_0}-principal", stop_event))
thread_1 = threading.Thread(target=video_thread, args=(source_1, f"Camera_{source_1}-secundaria", stop_event))

thread_position = threading.Thread(target=position_thread, args=(stop_event,))

thread_0.start()
thread_1.start()
thread_position.start()
 

try:
    # Aguarda o encerramento das threads
    thread_0.join()
    thread_1.join()
    thread_position.join()
except KeyboardInterrupt:
    stop_event.set()
    thread_0.join()
    thread_1.join()
    thread_position.join()


cv2.destroyAllWindows()
