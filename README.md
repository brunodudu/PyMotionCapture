# VideoTracker

Requisitos:
- [Roboflow Inference Docker](https://inference.roboflow.com/quickstart/docker/) local.
- `pip install inference-sdk`
- `pip install opencv-python`
- `pip install numpy`
- Também é necessário criar um arquivo `parameters.json` com a URL do seu Roboflow Inference Docker e sua API_KEY. Exemplo: `{"api_key": "qwerty12345","api_url": "http://localhost:9001"}`

## [Calib-1](Calib-1)
Faz a calibração das câmeras individualmente. 

Parâmetro: *{ID da webcam} / {url do stream de video} / {arquivo de video}*
- [get_chess_frames.py](Calib-1/get_chess_frames.py): Captura os frames ao se pressionar *s*. Deve-se filmar [pattern_chessboard.png](Calib-1/pattern_chessboard.png) de vários angulos. São necessários pelo menos **3** frames funcionais. Os frames são salvos em **Calib-1/frames**.

- [calib_cam.py](Calib-1/calib_cam.py): Faz a calibração usando os frames capturados anteriormente. Os resultados são salvos em **Calib-1/results**.

## [Calib-2](Calib-2)
Calibra o par de câmeras encontrando a posição e a matriz rotação da câmera secundária em relação a principal. 
- [cap_8+_calib.py](Calib-2/cap_8+_calib.py): Captura as posições do objeto detectado ao se pressionar *s* e salva em **Calib-2/results**. É necessário que o objeto esteja detectado nas duas câmeras. São necessários pelo menos **8** frames em posições variadas.

Parâmetros:
1. *{ID da webcam principal} / {url do stream de video principal} / {arquivo de video principal}*
2. *{ID da webcam secundária} / {url do stream de video secundário} / {arquivo de video secundário}*

- [calib_Rt.py](Calib-2/calib_Rt.py): Usa as posições detectadas anteriormente para calcular a posição e a matriz rotação da câmera secundária em relação a principal e salva em **Calib-2/results**.

Parâmetros:
1. *{ID da webcam principal} / {url do stream de video principal} / {arquivo de video principal}*
2. *{ID da webcam secundária} / {url do stream de video secundário} / {arquivo de video secundário}*
3. *{Distância entre as câmeras na unidade de medida desejada}*

## [video_tracker](video_tracker.py)
Script principal.

Parâmetros:
1. *{ID da webcam principal} / {url do stream de video principal} / {arquivo de video principal}*
2. *{ID da webcam secundária} / {url do stream de video secundário} / {arquivo de video secundário}*

Exemplo: `python video_tracker.py 1 0` - webcam *1* como principal e *0* como secundária.