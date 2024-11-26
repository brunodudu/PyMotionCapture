import cv2
import threading
from inference_sdk import InferenceHTTPClient
import numpy as np
import argparse
import json
import os

# Configuração do argparse para receber parâmetros da linha de comando
parser = argparse.ArgumentParser(description="Extrair pelo menos 8 pares de imagens detectando objeto em duas cameras.")
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

project_id = "balls-obj-det-3.0-euven"
model_version = 2
api_key = "S3lt1G4Wx3nBEACDxu9z"
api_url = "http://localhost:9001"

client = InferenceHTTPClient(api_url=api_url, api_key=api_key)

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
        elif key & 0xFF == ord('s'):
            if string_display in result_threads:
                results_shots.append(result_threads.copy())  # Salva o ponto (x, y) na lista correspondente
                print(results_shots) 

    
    cap.release()
    cv2.destroyWindow(string_display)


# Evento para parar as threads
stop_event = threading.Event()

result_threads = dict()
results_shots = []

# Iniciando as threads para as câmeras
thread_0 = threading.Thread(target=video_thread, args=(source_0, f"Camera_{source_0}", stop_event))
thread_1 = threading.Thread(target=video_thread, args=(source_1, f"Camera_{source_1}", stop_event))

thread_0.start()
thread_1.start()
 

try:
    # Aguarda o encerramento das threads
    thread_0.join()
    thread_1.join()
except KeyboardInterrupt:
    stop_event.set()
    thread_0.join()
    thread_1.join()

print(results_shots)
os.makedirs("results", exist_ok=True)
with open(f"results/shots_for_FundamentalMatrix.json", "w") as json_file:
    json.dump(results_shots, json_file)

cv2.destroyAllWindows()
