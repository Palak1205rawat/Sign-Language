from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import cv2
import numpy as np
# import mediapipe as mp
import os

app = Flask(__name__)
CORS(app)

# ✅ GLOBAL STATE
running = False

# ✅ Debug: check files on Render
print("📂 Files in directory:", os.listdir())

model_path = "sign_language_model.p"

if not os.path.exists(model_path):
    raise Exception("❌ Model file NOT found")

bundle = pickle.load(open(model_path, "rb"))

# 🔹 Load model
#bundle = pickle.load(open("sign_language_model.p", "rb"))
model = bundle["model"]
le = bundle["label_encoder"]
print("✅ Model loaded successfully")

# 🔹 MediaPipe setup (optimized)
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=1)
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     max_num_hands=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )
# from mediapipe.python.solutions import hands as mp_hands

# hands = mp_hands.Hands(
#     max_num_hands=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )

# print("✅ MediaPipe initialized")

# ✅ Health check (important for Render)
@app.route("/")
def home():
    return "Gesture API is running successfully"

# ▶️ Start
@app.route("/start", methods=["GET"])
def start():
    global running
    running = True
    print("🟢 Gesture Started")
    return jsonify({"status": "started"})

# ⏹️ Stop
@app.route("/stop", methods=["GET"])
def stop():
    global running
    running = False
    print("🔴 Gesture Stopped")
    return jsonify({"status": "stopped"})

# 🔥 Predict
@app.route("/predict", methods=["POST"])
def predict():
    global running

    if not running:
        return jsonify({"label": ""})

    if "frame" not in request.files:
        return jsonify({"label": "No frame received"})

    try:
        file = request.files["frame"]

        img = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"label": "Invalid frame"})

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        label = "No hand detected"

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                data = []
                for lm in handLms.landmark:
                    data.extend([lm.x, lm.y, lm.z])

                if len(data) == 63:
                    pred = model.predict([data])
                    label = str(le.inverse_transform(pred)[0])

        return jsonify({"label": label})

    except Exception as e:
        print("Error:", e)
        return jsonify({"label": "Error"})
# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     global running

#     if not running:
#         return jsonify({"label": ""})

#     return jsonify({"label": "Gesture coming from frontend"})
# @app.route("/predict", methods=["POST"])
# def predict():
#     global running

#     print("📥 Request received")

#     if not running:
#         return jsonify({"label": ""})

#     if "frame" not in request.files:
#         return jsonify({"label": "No frame received"})

#     file = request.files["frame"]

#     try:
#         # Convert image
#         img = np.frombuffer(file.read(), np.uint8)
#         frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

#         if frame is None:
#             return jsonify({"label": "Invalid frame"})

#         frame = cv2.flip(frame, 1)
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         result = hands.process(rgb)

#         label = "No hand detected"

#         if result.multi_hand_landmarks:
#             for handLms in result.multi_hand_landmarks:
#                 data = []

#                 for lm in handLms.landmark:
#                     data.extend([lm.x, lm.y, lm.z])

#                 if len(data) == 63:
#                     pred = model.predict([data])
#                     label = str(le.inverse_transform(pred)[0])
#                     print("🤟 Prediction:", label)

#         return jsonify({"label": label})

#     except Exception as e:
#         print("❌ Error:", str(e))
#         return jsonify({"label": "Error"})

# ✅ Render compatible run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)

# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import pickle
# import cv2
# import numpy as np
# import mediapipe as mp
# import os 

# app = Flask(__name__)
# CORS(app)  # ✅ allow requests from localhost:3000

# # 🔹 Load model
# bundle = pickle.load(open("sign_language_model.p", "rb"))
# model = bundle["model"]
# le = bundle["label_encoder"]

# # 🔹 MediaPipe setup
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=1)

# # cap = None
# # running = False

# @app.route("/")
# def home():
#     return "Gesture API is running successfully"


# # ▶️ Start gesture detection
# @app.route("/start")
# def start():
#     global running
#     running = True
#     return jsonify({"status": "started"})

# # @app.route("/start")
# # def start():
# #     global cap, running
# #     cap = cv2.VideoCapture(0)
# #     running = True
# #     return jsonify({"status": "started"})

# # ⏹️ Stop gesture detection
# @app.route("/stop")
# def stop():
#     global running
#     running = False
#     return jsonify({"status": "stopped"})

# # @app.route("/stop")
# # def stop():
# #     global cap, running
# #     running = False
# #     if cap:
# #         cap.release()
# #     return jsonify({"status": "stopped"})


# # 🔥 MAIN FIX: receive frame from frontend
# @app.route("/predict", methods=["POST"])
# def predict():
#     global running

#     if not running:
#         return jsonify({"label": ""})

#     if "frame" not in request.files:
#         return jsonify({"label": ""})

#     file = request.files["frame"]

#     # convert image
#     img = np.frombuffer(file.read(), np.uint8)
#     frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     result = hands.process(rgb)

#     label = ""

#     if result.multi_hand_landmarks:
#         for handLms in result.multi_hand_landmarks:
#             data = []
#             for lm in handLms.landmark:
#                 data.extend([lm.x, lm.y, lm.z])

#             if len(data) == 63:
#                 pred = model.predict([data])
#                 label = str(le.inverse_transform(pred)[0])

#     return jsonify({"label": label})


# if __name__ == "__main__":
#     app.run(port=5001, debug=True)


