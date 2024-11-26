# Video acessivel em http://[ip_host]:5000/video

from flask import Flask, Response
import cv2

app = Flask(__name__)

# Função para capturar frames da webcam
def generate_frames():
    camera = cv2.VideoCapture(0)  # Acesse a webcam (0 para a primeira webcam conectada)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Codifica o frame em formato JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
