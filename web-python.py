from flask import Flask
from flask import render_template
from flask import Response
import cv2
import torch


app = Flask(__name__)
url = "rtsp://conexao_sua_camera"
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0] #só para pessoas
model.conf = 0.65 #confiança
# Inicia a captura de vídeo
cap = cv2.VideoCapture(url)
larguraVideo, alturaVideo = 800, 600  #Medidas câmera video
# larguraVideo, alturaVideo = 640, 480  #Medidas câmera video
def generate():
    while True:
        while(cap.isOpened()):
            # Lê um quadro do feed de vídeo
            ret, frame = cap.read()

            if not ret:
                print("Erro ao capturar o quadro do vídeo")
            results = model(frame)
            labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
            classes = "pessoa"
            n = len(labels)
            x_shape, y_shape = frame.shape[1], frame.shape[0]
            for i in range(n):
                row = cord[i]
                if row[4] >= 0.2:
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    bgr = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
                    i = int(i)
                    cv2.putText(frame, classes, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
            frame = cv2.resize(frame, (larguraVideo, alturaVideo))
            (flag, encodedImage) = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True, threaded=True, host="ip_servidor")