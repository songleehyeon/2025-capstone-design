# 모델 파일 경로
YOLO_MODEL_PATH = "models/yolov8n.pt"
# AGE_GENDER_MODEL_PATH = "models/age_gender_mobilenet.pth"
FACE_PROTO = "models/opencv_face_detector.pbtxt"
FACE_MODEL = "models/opencv_face_detector_uint8.pb"
AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

# 데모 영상 경로
DEMO_VIDEO_PATH = "assets/videos/street_demo.mp4"

# 데모 영상 경로
DEMO_VIDEO_PATH = "assets/videos/street_demo.mp4"

# 날씨 API
WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY" # [TODO]
LOCATION_CITY = "Seoul"

# 집계 설정
AGGREGATION_WINDOW_SIZE = 30 # 30프레임 (약 1초) 동안 데이터 집계