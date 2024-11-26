import cv2
import os
import argparse

# Configuração do argparse para receber parâmetros da linha de comando
parser = argparse.ArgumentParser(description="Extrair frames de um vídeo e salvar como JPG.")
parser.add_argument("source", type=str, help="Fonte do vídeo.")
args = parser.parse_args()

# Determinar se a fonte é um índice de webcam ou um arquivo de vídeo
try:
    source = int(args.source)  # Tenta converter para inteiro
except ValueError:
    source = args.source  # Se falhar, mantém como string (caminho do vídeo)

output_folder = f"frames_{source}"

# Criar a pasta para salvar os frames
os.makedirs(output_folder, exist_ok=True)

# Abrir o vídeo
cap = cv2.VideoCapture(source)
frame_number = 0

while True:
    ret, frame = cap.read()
    
    # Verificar se o frame foi capturado com sucesso
    if not ret:
        print("Erro ao capturar frame")
        break

    cv2.imshow(f"Camera {source}", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):

        # Nome do arquivo do frame
        frame_filename = os.path.join(output_folder, f'frame_{frame_number:04d}.jpg')

        # Salvar o frame como JPG
        cv2.imwrite(frame_filename, frame)
        
        print(f'Salvou: {frame_filename}')
        
        frame_number += 1

# Liberar o vídeo
cap.release()
cv2.destroyAllWindows()
print(f"Extração de frames concluída! Frames salvos na pasta: {output_folder}")
