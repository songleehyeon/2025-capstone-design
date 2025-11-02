import streamlit as st
import cv2
import numpy as np

# 설정 파일
from config import settings

# 헬퍼 모듈 임포트
from src.analysis.detector import PersonDetector
from src.analysis.classifier import DemographicClassifier
from src.analysis.aggregator import DataAggregator
from src.context.weather_manager import get_weather_context
from src.context.time_manager import get_time_context
from src.logic.ad_database import load_ad_database
from src.logic.selection_engine import AdSelectionEngine
from src.utils.drawing import draw_results

# --- 1. 초기 설정 (Streamlit 캐싱) ---
# @st.cache_resource: 모델처럼 무거운 객체를 로드할 때 사용
@st.cache_resource
def load_models():
    """AI 모델을 로드하고 캐시합니다."""
    detector = PersonDetector()
    classifier = DemographicClassifier()
    return detector, classifier

# @st.cache_data: DB, API 호출 등 데이터 자체를 캐시할 때 사용
@st.cache_data
def load_data():
    """광고 DB를 로드하고 캐시합니다."""
    ad_db = load_ad_database("config/ad_db.json")
    return ad_db

@st.cache_data
def fetch_context():
    """외부 컨텍스트(날씨, 시간)를 가져옵니다."""
    time_tag = get_time_context()
    weather_tag = get_weather_context(settings.WEATHER_API_KEY, settings.LOCATION_CITY)
    return [time_tag, weather_tag] # 태그 리스트로 반환


# --- 2. 객체 생성 ---
st.set_page_config(layout="wide", page_title="Smart Ad Demo")
st.title("🤖 실시간 유동인구 분석 기반 옥외광고 데모")

# 모델, 데이터 로드
detector, classifier = load_models()
ad_db = load_data()
context_tags = fetch_context() # [time_tag, weather_tag]

# 로직 객체 생성
ad_engine = AdSelectionEngine(ad_db)
aggregator = DataAggregator() # 데이터 집계기

# --- 3. Streamlit UI 레이아웃 설정 ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header("실시간 분석 영상")
    # 비디오 프레임이 출력될 자리
    video_placeholder = st.empty()

with col2:
    st.header("대시보드")
    
    # Context 정보
    with st.container(border=True):
        st.subheader("Context")
        st.info(f"시간: **{context_tags[0]}** |  날씨: **{context_tags[1]}**")
    
    # 통계 정보
    with st.container(border=True):
        st.subheader(f"Crowd Stats (Recent {settings.AGGREGATION_WINDOW_SIZE} frames)")
        # 실시간 차트가 그려질 자리
        stats_placeholder = st.empty()
    
    # 광고 송출
    with st.container(border=True):
        st.subheader("광고 송출")
        # 선정 이유가 표시될 자리
        ad_reason_placeholder = st.empty()
        # 광고 영상이 출력될 자리
        ad_video_placeholder = st.empty()

# --- 4. 비디오 스트리밍 및 추론 루프 ---
cap = cv2.VideoCapture(settings.DEMO_VIDEO_PATH)

# 광고 상태를 저장하여 동일한 광고가 반복 재생되지 않도록 함
current_ad_path = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # 영상이 끝나면 처음부터 다시 재생 (데모용 루프)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 1. [AI] 사람 감지 (YOLO)
    person_boxes = detector.detect_persons(frame)
    
    # 2. [AI] 연령/성별 분류 (CNN)
    demographics = classifier.classify_demographics(frame, person_boxes)
    
    # 3. [Logic] 데이터 집계
    aggregator.add_data(demographics)
    dominant_group, stats_dict = aggregator.get_dominant_group_and_stats()

    # 4. [Logic] 광고 선정
    # (dominant_group, context_tags)를 기반으로 광고 선정
    selected_ad_path, reason = ad_engine.select_ad(dominant_group, context_tags)

    # 5. [UI] 결과 시각화
    
    # 5-1. 분석 영상 업데이트
    output_frame = draw_results(frame, person_boxes, demographics)
    video_placeholder.image(output_frame, channels="BGR", use_column_width=True)
    
    # 5-2. 통계 대시보드 업데이트
    if stats_dict:
        stats_placeholder.bar_chart(stats_dict)
    else:
        stats_placeholder.write("No crowd detected.")

    # 5-3. 광고 화면 업데이트
    ad_reason_placeholder.info(f"선정 이유: **{reason}**")
    
    # 광고가 바뀌었을 때만 비디오를 새로 로드
    if selected_ad_path and selected_ad_path != current_ad_path:
        current_ad_path = selected_ad_path
        ad_video_placeholder.video(current_ad_path, loop=True, autoplay=True, muted=True)
    elif not selected_ad_path:
        ad_video_placeholder.empty() # 송출할 광고가 없으면 비움
        current_ad_path = None
        
    # [Optional] 데모 속도 조절
    # time.sleep(1/30) # 30fps

cap.release()