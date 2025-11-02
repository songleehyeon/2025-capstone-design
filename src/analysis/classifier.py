import cv2
import numpy as np
from config import settings # 설정 파일 임포트

class DemographicClassifier:
    
    # 모델 추론에 필요한 상수 (원본 app.py와 동일)
    MODEL_MEAN_VALUES = (104.0, 177.0, 123.0)
    AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    GENDER_LIST = ['Male', 'Female']
    FACE_CONF_THRESHOLD = 0.7 # 얼굴 인식 최소 신뢰도
    FACE_SIZE = (300, 300)
    AGE_GENDER_SIZE = (227, 227)

    def __init__(self):
        # [수정] PyTorch 모델 대신 OpenCV DNN 모델 로드
        try:
            self.FACE_NET = cv2.dnn.readNet(settings.FACE_MODEL, settings.FACE_PROTO)
            self.AGE_NET = cv2.dnn.readNet(settings.AGE_MODEL, settings.AGE_PROTO)
            self.GENDER_NET = cv2.dnn.readNet(settings.GENDER_MODEL, settings.GENDER_PROTO)
            
            print("OpenCV DNN Age/Gender/Face models loaded successfully.")
            
            # [Optional] CPU 백엔드 명시
            self.FACE_NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.AGE_NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.GENDER_NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            
            self.FACE_NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.AGE_NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.GENDER_NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
        except Exception as e:
            print(f"Error loading OpenCV DNN models: {e}")
            print("Please check model paths in 'config/setting.py'")
            # 모델 로드 실패 시 빈 모델 할당 (에러 방지)
            self.FACE_NET = None
            self.AGE_NET = None
            self.GENDER_NET = None

    def _get_face_box(self, person_image):
        """YOLO가 크롭한 '사람' 이미지에서 '얼굴'을 찾습니다."""
        
        frame_height = person_image.shape[0]
        frame_width = person_image.shape[1]
        
        # 1. 얼굴 인식을 위한 Blob 생성
        blob = cv2.dnn.blobFromImage(person_image, 1.0, self.FACE_SIZE, self.MODEL_MEAN_VALUES, swapRB=False)
        self.FACE_NET.setInput(blob)
        detections = self.FACE_NET.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.FACE_CONF_THRESHOLD:
                # 2. 얼굴 좌표 계산
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                
                # 3. 좌표 보정 및 얼굴 이미지 크롭
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)

                # 유효한 크기의 얼굴이 감지된 경우
                if x2 > x1 and y2 > y1:
                    face_crop = person_image[y1:y2, x1:x2]
                    return face_crop # 첫 번째 감지된 얼굴만 반환
                    
        return None # 얼굴 감지 실패

    def classify_demographics(self, frame, person_boxes):
        """
        감지된 사람들 영역을 잘라내어 연령/성별을 분류합니다.
        (YOLO -> Face Detect -> Age/Gender)
        """
        results = []
        if not person_boxes or self.AGE_NET is None or self.GENDER_NET is None or self.FACE_NET is None:
            return results

        for box in person_boxes:
            x1, y1, x2, y2 = box
            
            # 1. YOLO 박스로 '사람' 영역 크롭
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # 2. '사람' 영역에서 '얼굴' 찾기
            face_crop = self._get_face_box(person_crop)
            
            # 3. 얼굴이 감지된 경우에만 나이/성별 추론
            if face_crop is not None and face_crop.size > 0:
                try:
                    # 4. 나이/성별 추론을 위한 Blob 생성
                    blob = cv2.dnn.blobFromImage(face_crop, 1.0, self.AGE_GENDER_SIZE, self.MODEL_MEAN_VALUES, swapRB=False)
                    
                    # 5. 성별 추론
                    self.GENDER_NET.setInput(blob)
                    gender_preds = self.GENDER_NET.forward()
                    gender = self.GENDER_LIST[gender_preds[0].argmax()]
                    
                    # 6. 나이 추론
                    self.AGE_NET.setInput(blob)
                    age_preds = self.AGE_NET.forward()
                    age_category = self.AGE_LIST[age_preds[0].argmax()]

                    # [TODO] 7. 태그 형식 변환 (DB와 맞추기)
                    # (e.g., age_category '(25-32)'와 gender 'Female'을 '20s_female'로)
                    # 이 부분은 ad_db.json의 태그 형식에 맞게 변환해야 합니다.
                    # 예시:
                    if age_category in ['(15-20)', '(25-32)']:
                        age_tag = "20s"
                    elif age_category in ['(38-43)', '(48-53)']:
                        age_tag = "30-50s"
                    else:
                        age_tag = "other" # 또는 age_category
                    
                    gender_tag = gender.lower() # "Female" -> "female"
                    
                    # ad_db.json 태그 형식: "20s_female", "30-50s_male"
                    final_tag = f"{age_tag}_{gender_tag}" 
                    
                    # 'other' 태그는 집계에서 제외하거나 따로 처리
                    if age_tag != "other":
                         results.append(final_tag)
                         
                except cv2.error as e:
                    # 크롭된 이미지가 너무 작거나 유효하지 않을 때 발생
                    # print(f"OpenCV error during inference: {e}")
                    continue
            
        return results