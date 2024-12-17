import cv2
import threading
from inference_sdk import InferenceHTTPClient
import numpy as np
import argparse
import json

with open("parameters.json", "r") as json_file:
    parameters = json.load(json_file)
    
project_id = "balls-obj-det-3.0-euven"
model_version = 2
api_key = parameters["api_key"]
api_url = parameters["api_url"]

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
    K_0 = np.array(json.load(json_file), dtype=np.float64)

with open(f"Calib-1/results/camera_matrix_{source_1}.json", "r") as json_file:
    K_1 = np.array(json.load(json_file), dtype=np.float64)

R_0 = np.eye(3)
t_0 = np.zeros(3)

with open(f"Calib-2/results/RotationMatrix.json", "r") as json_file:
    R_1 = np.array(json.load(json_file), dtype=np.float64)

with open(f"Calib-2/results/PositionVector.json", "r") as json_file:
    t_1 = np.array(json.load(json_file), dtype=np.float64)

K_0_inv = np.linalg.inv(K_0)
K_1_inv = np.linalg.inv(K_1)

def ponto_medio_retas(reta1, reta2):
    p_a = reta1[0]
    p_b = reta2[0]
    v_a = reta1[1]
    v_b = reta2[1]
    v_axb = np.cross(v_a, v_b)

    p = p_b - p_a
    S = np.column_stack((v_a, - v_b, v_axb))
    lamb = np.linalg.solve(S, p)
    
    a = p_a + (lamb[0] * v_a)
    b = p_b + (lamb[1] * v_b)
    return (a + b)/2

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
    global reproj

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

        if reproj[string_display][2] != 0:
            desenhar_centro(image, int(reproj[string_display][0]), int(reproj[string_display][1]), (255, 0, 0))
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
    global reproj
    while not stop_event.is_set():
        pixel_0 = result_threads[f"Camera_{source_0}-principal"]
        pixel_1 = result_threads[f"Camera_{source_1}-secundaria"]
        reproj_0 = np.zeros(3)
        reproj_1 = np.zeros(3)
        if pixel_0 != (0, 0) and pixel_1 != (0, 0):
            reta0 = reta3D(K_0_inv, np.transpose(R_0), t_0, pixel_0)
            reta1 = reta3D(K_1_inv, np.transpose(R_1), t_1, pixel_1)
            rec = ponto_medio_retas(reta0, reta1)
            reproj_0 = K_0 @ (np.concatenate((R_0, t_0[:, np.newaxis]), axis=1) @ np.transpose([rec[0], rec[1], rec[2], 1]))
            reproj_0 = reproj_0 / reproj_0[2]
            reproj_1 = K_1 @ (np.concatenate((R_1, t_1[:, np.newaxis]), axis=1) @ np.transpose([rec[0], rec[1], rec[2], 1]))
            reproj_1 = reproj_1 / reproj_1[2]
            reproj[f"Camera_{source_0}-principal"] = reproj_0
            reproj[f"Camera_{source_1}-secundaria"] = reproj_1
            print(f"Ponto:{rec}, Reproj_{source_0}:{reproj_0}, Reproj_{source_1}:{reproj_1}")
        else:
            reproj[f"Camera_{source_0}-principal"] = np.zeros(3)
            reproj[f"Camera_{source_1}-secundaria"] = np.zeros(3)

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
reproj = {
    f"Camera_{source_0}-principal": (0,0,0),
    f"Camera_{source_1}-secundaria": (0,0,0)
}

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
