import os
import sounddevice as sd  # type: ignore
import numpy as np  # type: ignore
import queue
from vosk import Model, KaldiRecognizer  # type: ignore
from flask import Flask, render_template  # type: ignore
from flask_socketio import SocketIO  # type: ignore
import threading
import json

# 設定
SAMPLERATE = 44100  # サンプリングレート
BUFFER_SIZE = 2048  # バッファサイズ（チャンクサイズ）

app = Flask(__name__)
socketio = SocketIO(app)

# Voskモデルの読み込み
model_path = "model"  #絶対パスを指定
if not os.path.exists(model_path):
    print(f"モデルが見つかりません: {model_path}")
    exit(1)

model = Model(model_path)
recognizer = KaldiRecognizer(model, SAMPLERATE)

# データ用キュー
q = queue.Queue()

# コールバック関数
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Error: {status}")
    q.put(indata.copy())

# 音声認識処理
def recognize_audio():
    try:
        # PCM データを WAV フォーマットに変換
        audio_data = []
        while not q.empty():
            audio_data.append(q.get())

        if len(audio_data) > 0:
            audio_data = np.concatenate(audio_data, axis=0)
            # PCM 16bit に変換
            audio_data = (audio_data * 32767).astype(np.int16)
            pcm_data = audio_data.tobytes()

            if recognizer.AcceptWaveform(pcm_data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    print(f"認識結果: {text}")
                    socketio.emit('new_text', text.split())  # 単語単位でクライアントに送信

    except Exception as e:
        print(f"音声認識中にエラーが発生しました: {e}")

# 音声ストリーミング開始
def start_audio_stream():
    stream = sd.InputStream(
        samplerate=SAMPLERATE,
        blocksize=BUFFER_SIZE,
        channels=1,
        callback=audio_callback
    )

    print("Recording...")
    stream.start()

    # 音声認識処理をバックグラウンドで開始
    while True:
        recognize_audio()

# Flaskルート
@app.route('/')
def index():
    return render_template('index.html')

# 停止用のルートを追加
@app.route('/stop', methods=['POST'])
def stop_server():
    """サーバーを停止させる"""
    print("サーバー停止要求を受け取りました...")
    os._exit(0)  # サーバーを終了

# サーバー起動
if __name__ == '__main__':
    # 音声ストリーミングを別スレッドで実行
    audio_thread = threading.Thread(target=start_audio_stream)
    audio_thread.daemon = True
    audio_thread.start()

    # Flaskサーバーを開始
    socketio.run(app, debug=True)
