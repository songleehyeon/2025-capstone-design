"""
OCR 텍스트 인식 모듈
EasyOCR, PaddleOCR, Tesseract를 활용하여 광고 영역에서 텍스트를 추출합니다.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import os
import json

# PaddleOCR 한글 경로 문제 해결: HOME 디렉토리를 영문 경로로 변경 (import 전에 설정)
os.environ['HOME'] = 'C:/PaddleOCR_models'
os.environ['USERPROFILE'] = 'C:/PaddleOCR_models'
os.environ['PADDLEX_HOME'] = 'C:/PaddleOCR_models'
os.environ['PPOCR_HOME'] = 'C:/PaddleOCR_models'
os.environ['HF_HOME'] = 'C:/PaddleOCR_models'

# 디렉토리 생성
os.makedirs('C:/PaddleOCR_models', exist_ok=True)


class OCRProcessor:
    """OCR 기반 텍스트 인식 및 처리"""

    def __init__(self, engine='easyocr', languages=['ko', 'en'], gpu=False, enable_llm_correction=False):
        """
        초기화

        Args:
            engine: OCR 엔진 ('easyocr', 'paddleocr', 'windows')
            languages: 인식할 언어 리스트
            gpu: GPU 사용 여부
            enable_llm_correction: LLM 기반 오타 교정 활성화
        """
        self.engine = engine.lower()
        self.languages = languages
        self.enable_llm_correction = enable_llm_correction

        print(f"OCRProcessor 초기화 중... (엔진: {self.engine.upper()}, GPU: {gpu}, LLM 교정: {enable_llm_correction})")

        if self.engine == 'easyocr':
            import easyocr
            # EasyOCR 초기화
            self.reader = easyocr.Reader(languages, gpu=gpu)
            print("✓ OCRProcessor 초기화 완료 (EasyOCR)")

        elif self.engine == 'paddleocr':
            from paddleocr import PaddleOCR
            # PaddleOCR 초기화
            self.reader = PaddleOCR(
                use_angle_cls=False,
                lang='korean'  # 한국어+영어 모델
            )
            print("✓ OCRProcessor 초기화 완료 (PaddleOCR - Korean)")

        elif self.engine == 'windows':
            try:
                # Windows OCR (winrt 필요, Python 3.11 이하에서만 작동)
                from winrt.windows.media.ocr import OcrEngine
                from winrt.windows.globalization import Language
                from winrt.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat

                # 한국어 언어 설정
                lang_code = 'ko' if 'ko' in languages else 'en'
                language = Language(lang_code)

                # OCR 엔진 생성
                self.reader = OcrEngine.try_create_from_language(language)

                if self.reader is None:
                    raise RuntimeError(f"Windows OCR 언어 팩이 설치되지 않음: {lang_code}")

                self.windows_lang = lang_code
                print(f"✓ OCRProcessor 초기화 완료 (Windows OCR, 언어: {lang_code})")
            except ImportError:
                raise RuntimeError(
                    "Windows OCR 사용 불가: winrt 라이브러리가 필요합니다.\n"
                    "Python 3.11 이하 환경에서 'pip install winrt'를 실행하세요.\n"
                    "참고: Python 3.12+ 에서는 winrt가 지원되지 않습니다."
                )
            except Exception as e:
                raise RuntimeError(f"Windows OCR 초기화 실패: {e}")

        else:
            raise ValueError(f"지원하지 않는 OCR 엔진: {engine}. 'easyocr', 'paddleocr', 'windows' 중 선택하세요.")

        # 성능 최적화를 위한 캐싱
        self.last_result = None
        self.last_image_hash = None
        self.frame_skip_count = 0
        self.skip_threshold = 3  # 3프레임마다 1회 OCR 실행

        # 텍스트 인식 통계
        self.total_recognitions = 0
        self.total_time = 0

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        OCR 정확도 향상을 위한 이미지 전처리

        Args:
            image: 원본 이미지

        Returns:
            전처리된 이미지
        """
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray)

        # 대비 향상 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 적응형 이진화
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        return binary

    def recognize_text(self, image: np.ndarray, use_preprocessing=False,
                       min_confidence=0.3) -> List[Dict]:
        """
        이미지에서 텍스트 인식

        Args:
            image: 입력 이미지
            use_preprocessing: 전처리 사용 여부
            min_confidence: 최소 신뢰도 임계값

        Returns:
            인식된 텍스트 정보 리스트
            [{'text': str, 'confidence': float, 'bbox': [[x,y], ...], 'center': [x,y]}, ...]
        """
        # 성능 최적화: 프레임 스킵
        self.frame_skip_count += 1
        if self.frame_skip_count < self.skip_threshold and self.last_result is not None:
            return self.last_result

        self.frame_skip_count = 0

        # 전처리 (선택적)
        if use_preprocessing:
            processed_image = self.preprocess_image(image)
        else:
            processed_image = image

        # OCR 실행
        start_time = time.time()
        parsed_results = []

        try:
            if self.engine == 'easyocr':
                # EasyOCR: readtext() 메서드
                # 반환 형식: [(bbox, text, confidence), ...]
                results = self.reader.readtext(processed_image)

                for bbox, text, confidence in results:
                    if confidence >= min_confidence:
                        # Bounding box 중심 계산
                        bbox_array = np.array(bbox)
                        center = bbox_array.mean(axis=0).tolist()

                        parsed_results.append({
                            'text': text,
                            'confidence': float(confidence),
                            'bbox': bbox,
                            'center': center
                        })

            elif self.engine == 'paddleocr':
                # PaddleOCR: ocr() 메서드
                # 최신 버전은 딕셔너리 형식으로 결과 반환
                results = self.reader.ocr(processed_image)

                # results가 None이거나 빈 리스트인 경우 처리
                if results is None or len(results) == 0 or results[0] is None:
                    return []

                # 첫 번째 페이지 결과 (딕셔너리)
                result_dict = results[0]

                # 딕셔너리에서 필요한 정보 추출
                if isinstance(result_dict, dict):
                    rec_texts = result_dict.get('rec_texts', [])
                    rec_scores = result_dict.get('rec_scores', [])
                    rec_polys = result_dict.get('rec_polys', [])

                    # 텍스트, 점수, 좌표가 모두 있는 경우
                    for i in range(len(rec_texts)):
                        if i < len(rec_scores) and i < len(rec_polys):
                            text = rec_texts[i]
                            confidence = rec_scores[i]
                            bbox = rec_polys[i]

                            if confidence >= min_confidence:
                                # Bounding box 중심 계산
                                bbox_array = np.array(bbox)
                                center = bbox_array.mean(axis=0).tolist()

                                parsed_results.append({
                                    'text': text,
                                    'confidence': float(confidence),
                                    'bbox': bbox.tolist() if isinstance(bbox, np.ndarray) else bbox,
                                    'center': center
                                })

        except Exception as e:
            print(f"OCR 오류 ({self.engine.upper()}): {e}")
            return []

        elapsed_time = time.time() - start_time
        self.total_recognitions += 1
        self.total_time += elapsed_time

        # LLM 기반 오타 교정 (선택적)
        if self.enable_llm_correction and len(parsed_results) > 0:
            parsed_results = self.correct_text_with_llm(parsed_results)

        # 결과 캐싱
        self.last_result = parsed_results

        return parsed_results

    def correct_text_with_llm(self, ocr_results: List[Dict]) -> List[Dict]:
        """
        LLM을 사용하여 OCR 결과의 오타 교정

        Args:
            ocr_results: OCR 결과 리스트

        Returns:
            교정된 OCR 결과 리스트
        """
        if not self.enable_llm_correction or len(ocr_results) == 0:
            return ocr_results

        try:
            from openai import OpenAI

            # OpenAI API 키 로드
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                # config 파일에서 읽기 시도
                config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'api_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        api_key = config.get('openai_api_key')

            if not api_key:
                print("경고: OpenAI API 키가 설정되지 않았습니다. LLM 교정을 건너뜁니다.")
                return ocr_results

            client = OpenAI(api_key=api_key)

            # OCR 텍스트 추출
            texts = [item['text'] for item in ocr_results]
            combined_text = ' '.join(texts)

            # LLM 프롬프트 구성
            prompt = f"""다음은 OCR로 인식한 한국어 광고 텍스트입니다. 글자 단위의 오타가 있을 수 있으니 문맥을 고려하여 교정해주세요.

원본 텍스트:
{combined_text}

요구사항:
1. 각 단어/구절의 오타를 문맥에 맞게 교정
2. 원본 텍스트의 순서와 구조 유지
3. 불필요한 설명 없이 교정된 텍스트만 반환
4. 교정된 각 텍스트를 JSON 배열로 반환 (원본과 같은 순서)

형식 예시:
["교정된 텍스트1", "교정된 텍스트2", ...]"""

            # API 호출
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # 비용 효율적인 모델
                messages=[
                    {"role": "system", "content": "당신은 OCR 텍스트 교정 전문가입니다. 광고 및 메뉴판의 한국어 텍스트를 정확하게 교정합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 일관성 있는 교정
                max_tokens=500
            )

            # 응답 파싱
            corrected_text = response.choices[0].message.content.strip()

            # JSON 배열로 파싱 시도
            try:
                corrected_texts = json.loads(corrected_text)

                # 교정된 텍스트로 OCR 결과 업데이트
                corrected_results = []
                for i, item in enumerate(ocr_results):
                    corrected_item = item.copy()
                    if i < len(corrected_texts):
                        corrected_item['text'] = corrected_texts[i]
                        corrected_item['original_text'] = item['text']  # 원본 보존
                        corrected_item['llm_corrected'] = True
                    corrected_results.append(corrected_item)

                print(f"✓ LLM 교정 완료: {len(corrected_results)}개 텍스트")
                return corrected_results

            except json.JSONDecodeError:
                # JSON 파싱 실패 시 원본 반환
                print(f"경고: LLM 응답 파싱 실패. 원본 텍스트 사용.")
                return ocr_results

        except ImportError:
            print("경고: openai 라이브러리가 설치되지 않았습니다. 'pip install openai'를 실행하세요.")
            return ocr_results
        except Exception as e:
            print(f"LLM 교정 오류: {e}")
            return ocr_results

    def draw_text_boxes(self, image: np.ndarray, ocr_results: List[Dict],
                        color=(0, 255, 0), thickness=2) -> np.ndarray:
        """
        인식된 텍스트 영역을 이미지에 시각화

        Args:
            image: 원본 이미지
            ocr_results: OCR 결과 리스트
            color: 박스 색상 (B, G, R)
            thickness: 선 두께

        Returns:
            텍스트 박스가 그려진 이미지
        """
        result_image = image.copy()

        for item in ocr_results:
            bbox = item['bbox']
            text = item['text']
            confidence = item['confidence']

            # Bounding box 그리기
            points = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(result_image, [points], True, color, thickness)

            # 텍스트 및 신뢰도 표시
            label = f"{text} ({confidence:.2f})"
            label_pos = (int(bbox[0][0]), int(bbox[0][1]) - 10)

            # 배경 사각형
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                result_image,
                (label_pos[0], label_pos[1] - text_size[1] - 5),
                (label_pos[0] + text_size[0], label_pos[1] + 5),
                color,
                -1
            )

            # 텍스트
            cv2.putText(
                result_image,
                label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )

        return result_image

    def extract_keywords(self, ocr_results: List[Dict],
                         keyword_list: List[str]) -> Dict[str, List[Dict]]:
        """
        OCR 결과에서 특정 키워드 추출

        Args:
            ocr_results: OCR 결과 리스트
            keyword_list: 검색할 키워드 리스트

        Returns:
            키워드별 매칭된 텍스트 정보
        """
        keyword_matches = {keyword: [] for keyword in keyword_list}

        for result in ocr_results:
            text = result['text'].lower()

            for keyword in keyword_list:
                # 부분 문자열 매칭
                if keyword.lower() in text:
                    keyword_matches[keyword].append(result)

        return keyword_matches

    def get_all_text(self, ocr_results: List[Dict]) -> str:
        """
        모든 인식된 텍스트를 하나의 문자열로 합치기

        Args:
            ocr_results: OCR 결과 리스트

        Returns:
            통합된 텍스트 문자열
        """
        texts = [item['text'] for item in ocr_results]
        return ' '.join(texts)

    def get_statistics(self) -> Dict:
        """
        OCR 성능 통계 반환

        Returns:
            통계 정보 딕셔너리
        """
        avg_time = self.total_time / self.total_recognitions if self.total_recognitions > 0 else 0

        return {
            'total_recognitions': self.total_recognitions,
            'total_time': self.total_time,
            'average_time': avg_time,
            'fps': 1.0 / avg_time if avg_time > 0 else 0
        }

    def find_text_by_position(self, ocr_results: List[Dict],
                              position: Tuple[int, int],
                              radius: int = 50) -> Optional[Dict]:
        """
        특정 위치 근처의 텍스트 찾기 (터치 인터랙션용)

        Args:
            ocr_results: OCR 결과 리스트
            position: 검색 위치 (x, y)
            radius: 검색 반경

        Returns:
            가장 가까운 텍스트 정보 또는 None
        """
        min_distance = float('inf')
        closest_text = None

        for result in ocr_results:
            center = result['center']
            distance = np.linalg.norm(np.array(center) - np.array(position))

            if distance < radius and distance < min_distance:
                min_distance = distance
                closest_text = result

        return closest_text
