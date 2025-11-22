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
import json
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

# 초기화 상태 추적
backend_ready = False
initialization_status = {
    'ocr_ready': False,
    'personalization_ready': False,
    'aruco_ready': False
}

ocr_processor = OCRProcessor(engine='paddleocr', gpu=False, enable_llm_correction=False, use_aruco=True)
initialization_status['ocr_ready'] = True
initialization_status['aruco_ready'] = True

personalization_engine = PersonalizationEngine()
initialization_status['personalization_ready'] = True

backend_ready = True
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


@app.route('/3d-demo')
def demo_3d():
    """3D 광고판 시뮬레이터 (POV 방식)"""
    return render_template('3d_demo.html')


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
    (현재는 사용하지 않음 - 모든 광고는 설정 기반 통합 추천 엔진에서 처리)

    Args:
        ocr_results: OCR 결과 리스트

    Returns:
        하드코딩 룰로 생성된 오버레이 리스트
    """
    overlays = []

    # 모든 광고는 personalization_engine의 통합 추천 엔진에서 자동으로 처리됨
    # 브랜드별 룰은 config/ads/*.json 파일에 정의되어 있음

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
            "user_id": "user01",
            "gender": "female",
            "age": 28,
            "age_group": "20s",
            "occupation": ["worker", "bank"],
            "living_type": ["single_household", "single"],
            "allergies": [],
            "vegan": false,
            "low_sugar_preference": true,
            "low_caffeine_preference": false,
            "price_sensitivity": "high",
            "attribute_preferences": ["ice", "sweet", "latte", ...],
            "context": {
                "current_time": "afternoon",
                "day_type": "weekday",
                "weather": "cold"
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
        print(f"  성별: {data.get('gender', 'Unknown')}, 나이: {data.get('age', 'Unknown')}")
        print(f"  선호 속성: {data.get('attribute_preferences', [])}")

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


@app.route('/api/ready', methods=['GET'])
def check_readiness():
    """백엔드 준비 상태 확인"""
    return jsonify({
        'ready': backend_ready,
        'status': initialization_status,
        'message': '백엔드 준비 완료' if backend_ready else 'OCR 모델 로딩중...'
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'ocr_engine': ocr_processor.engine,
        'user_id': personalization_engine.user_profile.get('user_id', 'Unknown')
    })


@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """
    OCR 캐시 초기화

    초기화 버튼을 눌렀을 때 백엔드의 OCR 캐시를 완전히 제거
    """
    try:
        ocr_processor.clear_cache()
        return jsonify({
            'success': True,
            'message': 'OCR 캐시가 초기화되었습니다.'
        })
    except Exception as e:
        print(f"캐시 초기화 오류: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/personas', methods=['GET'])
def get_personas():
    """모든 페르소나 목록 조회"""
    try:
        personas_dir = Path(__file__).parent / 'config' / 'personas'
        personas = {}

        if not personas_dir.exists():
            print(f"경고: 페르소나 디렉토리를 찾을 수 없습니다: {personas_dir}")
            return jsonify({
                'success': True,
                'personas': {}
            })

        # personas/ 폴더의 모든 JSON 파일 로드
        for persona_file in personas_dir.glob("*.json"):
            try:
                with open(persona_file, 'r', encoding='utf-8') as f:
                    persona_data = json.load(f)
                    persona_id = persona_data.get('id')
                    if persona_id:
                        personas[persona_id] = persona_data
            except Exception as e:
                print(f"페르소나 파일 로드 오류 ({persona_file.name}): {e}")
                continue

        return jsonify({
            'success': True,
            'personas': personas
        })
    except Exception as e:
        print(f"페르소나 로드 오류: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def convert_persona_to_profile(persona_data: dict) -> dict:
    """
    페르소나 구조를 user_profile 구조로 변환

    Args:
        persona_data: personas.json의 페르소나 데이터

    Returns:
        user_profile.json 형식의 프로필
    """
    demographics = persona_data.get('demographics', {})
    living = persona_data.get('living', {})
    dietary = persona_data.get('dietary', {})
    preferences = persona_data.get('preferences', {})
    context = persona_data.get('context', {})

    profile = {
        'user_id': persona_data.get('id', 'unknown'),
        'persona_type': persona_data.get('id', 'unknown'),
        'persona_name': persona_data.get('displayName', persona_data.get('name', 'Unknown')),
        'gender': demographics.get('gender', 'unknown'),
        'age': demographics.get('age', 25),
        'age_group': demographics.get('age_group', '20s'),
        'occupation': demographics.get('occupation', []),
        'living_type': living.get('type', []),
        'allergies': dietary.get('allergies', []),
        'vegan': dietary.get('vegan', False),
        'low_sugar_preference': dietary.get('low_sugar_preference', False),
        'low_caffeine_preference': dietary.get('low_caffeine_preference', False),
        'price_sensitivity': preferences.get('price_sensitivity', 'medium'),
        'attribute_preferences': preferences.get('attribute_preferences', []),
        'context': context,
        'personalization_level': 'high'
    }

    return profile


@app.route('/api/select_persona', methods=['POST'])
def select_persona():
    """
    페르소나 선택 및 활성화

    Request:
        {
            "persona_id": "young-female"
        }

    Response:
        {
            "success": true,
            "profile": {...}
        }
    """
    try:
        data = request.get_json()
        persona_id = data.get('persona_id')

        if not persona_id:
            return jsonify({
                'success': False,
                'error': '페르소나 ID가 필요합니다.'
            }), 400

        # 페르소나 데이터 로드
        persona_file_path = Path(__file__).parent / 'config' / 'personas' / f'{persona_id}.json'

        if not persona_file_path.exists():
            return jsonify({
                'success': False,
                'error': f'페르소나를 찾을 수 없습니다: {persona_id}'
            }), 404

        with open(persona_file_path, 'r', encoding='utf-8') as f:
            persona_data = json.load(f)

        # 페르소나를 user_profile 형식으로 변환
        user_profile = convert_persona_to_profile(persona_data)

        # personalization_engine의 프로필 업데이트
        personalization_engine.user_profile = user_profile

        # user_profile.json 파일에 저장
        personalization_engine._save_profile()

        print(f"✓ 페르소나 선택: {user_profile.get('persona_name', 'Unknown')}")
        print(f"  ID: {persona_id}")

        return jsonify({
            'success': True,
            'message': f"페르소나 '{user_profile.get('persona_name')}' 선택됨",
            'profile': user_profile
        })

    except Exception as e:
        print(f"페르소나 선택 오류: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/recommended_ads', methods=['GET'])
def get_recommended_ads():
    """
    현재 페르소나에 따른 추천 광고 목록 반환

    Response:
        {
            "success": true,
            "recommended_ads": ["10", "9", "8", "12", ...]
        }
    """
    try:
        user_id = personalization_engine.user_profile.get('user_id', 'user00')

        # 페르소나별 추천 광고 매핑 (data-id 기준)
        recommendations = {
            'user00': ['10', '9', '8', '12'],  # 스타벅스, 서브웨이, RPG게임, 위키드2
            'user01': ['10', '9', '8', '11', '12', '7', '5', '6'],  # + 더현대서울, 헬스장, 헤라, 좋은데이
            'user02': ['10', '9', '8', '11', '12', '7', '6'],  # + 더현대서울, 헬스장, 좋은데이
            'user03': ['10', '9', '11', '12', '5', '6', '2'],  # + 더현대서울, 헤라, 좋은데이, 헤네시
            'user04': ['10', '9', '12', '6', '2'],  # 스타벅스, 서브웨이, 위키드2, 좋은데이, 헤네시
        }

        recommended = recommendations.get(user_id, [])

        return jsonify({
            'success': True,
            'recommended_ads': recommended,
            'user_id': user_id
        })

    except Exception as e:
        print(f"추천 광고 조회 오류: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'recommended_ads': []
        }), 500


if __name__ == '__main__':
    import socket

    # VPN이 있어도 실제 Wi-Fi IP를 정확하게 가져오기
    def get_local_ip():
        try:
            # 외부 연결을 시도해서 로컬 IP 가져오기 (실제로 연결하지는 않음)
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            # 실패 시 기본 방법 사용
            return socket.gethostbyname(socket.gethostname())

    local_ip = get_local_ip()

    print("\n🚀 Flask 서버 시작...")
    print(f"📱 PC 브라우저: https://localhost:5000")
    print(f"📱 스마트폰: https://{local_ip}:5000")
    print(f"\n⚠️  인증서 경고 해결 방법:")
    print(f"   1. 브라우저에서 '고급' 또는 'Advanced' 클릭")
    print(f"   2. '안전하지 않음(계속)' 또는 'Proceed to...' 클릭")
    print(f"   3. 카메라 권한 허용")
    print(f"\n💡 같은 Wi-Fi에 연결되어 있는지 확인하세요")
    print("\n종료하려면 Ctrl+C를 누르세요\n")

    # HTTPS로 실행 (카메라 접근 필수)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, ssl_context='adhoc')
