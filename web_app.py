"""
웹 기반 스마트 광고 시스템 - Flask 백엔드

Camo camera를 통한 웹캠 스트림을 받아 OCR 처리 후
개인화된 오버레이 데이터를 JSON으로 반환
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import sys
from pathlib import Path

# Windows 콘솔 인코딩 문제 해결
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# src 모듈 import를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ocr_processor import OCRProcessor
from personalization import PersonalizationEngine

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # CORS 허용 (프론트엔드와 백엔드가 다른 포트일 경우)

# 전역 모듈 초기화
print("=" * 60)
print("웹 기반 스마트 광고 시스템 초기화 중...")
print("=" * 60)

ocr_processor = OCRProcessor(engine='paddleocr', gpu=False, enable_llm_correction=False)
personalization_engine = PersonalizationEngine()

print("✓ 백엔드 초기화 완료")
print("=" * 60)


@app.route('/')
def index():
    """메인 페이지 (스마트폰 AR 뷰)"""
    return render_template('index.html')


@app.route('/admin')
def admin():
    """관리 페이지 (PC 페르소나 설정)"""
    return render_template('admin.html')


@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """
    프레임 처리 API

    Request:
        {
            "image": "base64_encoded_image"
        }

    Response:
        {
            "success": true,
            "ocr_results": [...],
            "overlays": [...]
        }
    """
    try:
        # Base64 이미지 받기
        data = request.get_json()
        image_data = data.get('image', '')

        if not image_data:
            return jsonify({
                'success': False,
                'error': '이미지 데이터가 없습니다.'
            }), 400

        # Base64 디코딩
        # "data:image/jpeg;base64," 프리픽스 제거
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                'success': False,
                'error': '이미지 디코딩 실패'
            }), 400

        # OCR 처리
        ocr_results = ocr_processor.recognize_text(frame, use_preprocessing=False)

        # 콘텐츠 분석
        content_analysis = personalization_engine.analyze_text_content(ocr_results)

        # 개인화된 오버레이 생성
        overlays = personalization_engine.generate_personalized_content(content_analysis)

        # 하드코딩 룰 적용
        hardcoded_overlays = apply_hardcoded_rules(ocr_results)
        overlays.extend(hardcoded_overlays)

        # 응답
        return jsonify({
            'success': True,
            'ocr_results': ocr_results,
            'overlays': overlays,
            'frame_size': {
                'width': frame.shape[1],
                'height': frame.shape[0]
            }
        })

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def apply_hardcoded_rules(ocr_results):
    """
    하드코딩된 룰 기반 오버레이 생성

    Args:
        ocr_results: OCR 결과 리스트

    Returns:
        하드코딩 룰로 생성된 오버레이 리스트
    """
    overlays = []

    # 모든 텍스트 합치기 (대소문자 구분 없이)
    all_text = ' '.join([item['text'] for item in ocr_results])
    all_text_lower = all_text.lower()

    # ========== 테스트용 광고 전단지 룰 ==========

    # 룰 1: "LG" 또는 "electronics" 감지 시
    if 'lg' in all_text_lower or 'electronics' in all_text_lower or 'electron' in all_text_lower:
        overlays.append({
            'type': 'promotion',
            'position': 'top-center',
            'content': '🔌 LG 신제품 출시 특가! 최대 30% 할인',
            'style': 'coffee',  # 파란색 계열로 변경 가능
            'reason': '하드코딩 룰: LG Electronics 감지'
        })
        overlays.append({
            'type': 'badge',
            'position': 'top-right',
            'content': '5년 무상 A/S',
            'style': 'new',
            'reason': '하드코딩 룰: LG 부가 혜택'
        })

    # 룰 2: "BALANCE" 또는 "LAB" 감지 시
    if 'balance' in all_text_lower or 'lab' in all_text_lower or '근골격' in all_text or '운동' in all_text:
        overlays.append({
            'type': 'promotion',
            'position': 'center',
            'content': '💪 첫 방문 고객 1개월 무료 체험!',
            'style': 'dessert',
            'reason': '하드코딩 룰: BALANCE LAB 감지'
        })
        overlays.append({
            'type': 'suggestion',
            'position': 'bottom-center',
            'content': '🏋️ PT 3개월 등록 시 1개월 추가 증정',
            'style': 'dessert',
            'reason': '하드코딩 룰: BALANCE LAB 프로모션'
        })

    # ========== 기존 카페 관련 룰 (참고용) ==========

    # 룰 3: "라떼" 감지 시 따뜻한 음료 추천
    if '라떼' in all_text or '라떼' in all_text:
        overlays.append({
            'type': 'promotion',
            'position': 'top-left',
            'content': '☕ 쌀쌀한 아침에 따뜻한 라떼는 어떠세요?',
            'style': 'coffee',
            'reason': '하드코딩 룰: 라떼 감지'
        })

    # 룰 4: "아메리카노" 감지 시 추가 샷 추천
    if '아메리카노' in all_text:
        overlays.append({
            'type': 'badge',
            'position': 'top-right',
            'content': '샷 추가 +500원',
            'style': 'espresso',
            'reason': '하드코딩 룰: 아메리카노 감지'
        })

    # 룰 5: "케이크" 또는 "디저트" 감지 시 페어링 추천
    if '케이크' in all_text or '디저트' in all_text:
        overlays.append({
            'type': 'suggestion',
            'position': 'bottom-left',
            'content': '🍰 커피와 함께 즐기시면 더 맛있어요!',
            'style': 'dessert',
            'reason': '하드코딩 룰: 디저트 감지'
        })

    return overlays


@app.route('/api/user_profile', methods=['GET'])
def get_user_profile():
    """사용자 프로필 조회"""
    return jsonify({
        'success': True,
        'profile': personalization_engine.user_profile
    })


@app.route('/api/update_profile', methods=['POST'])
def update_profile():
    """
    사용자 프로필 업데이트

    Request:
        {
            "user_id": "user001",
            "preferences": {
                "coffee_type": ["라떼", "아메리카노"],
                "dietary": {
                    "allergies": ["우유"],
                    "vegetarian": false,
                    "vegan": false
                },
                "price_sensitivity": "medium",
                "interests": ["패션", "IT"]
            },
            "personalization_level": "high"
        }

    Response:
        {
            "success": true,
            "message": "프로필 업데이트 완료"
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': '데이터가 없습니다.'
            }), 400

        # 프로필 업데이트
        personalization_engine.user_profile = data

        # 파일에 저장
        personalization_engine._save_profile()

        print(f"✓ 프로필 업데이트 완료: {data.get('user_id', 'Unknown')}")
        print(f"  선호 음료: {data.get('preferences', {}).get('coffee_type', [])}")
        print(f"  관심사: {data.get('preferences', {}).get('interests', [])}")

        return jsonify({
            'success': True,
            'message': '프로필 업데이트 완료',
            'profile': personalization_engine.user_profile
        })

    except Exception as e:
        print(f"프로필 업데이트 오류: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'ocr_engine': ocr_processor.engine,
        'user_id': personalization_engine.user_profile.get('user_id', 'Unknown')
    })


if __name__ == '__main__':
    print("\n🚀 Flask 서버 시작...")
    print("📱 브라우저에서 https://localhost:5000 접속 (HTTPS)")
    print("💡 스마트폰에서도 접속 가능 (HTTPS 필수)")
    print("\n종료하려면 Ctrl+C를 누르세요\n")

    # HTTPS로 실행 (adhoc SSL 사용 - 개발/테스트용)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, ssl_context='adhoc')
