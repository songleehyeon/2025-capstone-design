from collections import deque, Counter
from config import settings

class DataAggregator:
    """
    최근 N 프레임 동안의 인구 통계 데이터를 집계하여
    가장 많은 그룹(dominant group)과 전체 통계를 계산합니다.
    """
    def __init__(self):
        # 설정 파일에서 정의한 윈도우 크기만큼 큐를 생성
        self.queue = deque(maxlen=settings.AGGREGATION_WINDOW_SIZE)

    def add_data(self, demographics_list):
        """
        현재 프레임에서 감지된 인구 통계 리스트(e.g., ["20s_female", "40s_male"])를
        큐에 추가합니다.
        """
        self.queue.append(demographics_list)

    def get_dominant_group_and_stats(self):
        """
        큐에 쌓인 모든 데이터를 취합하여 가장 많은 그룹과
        전체 통계 딕셔너리를 반환합니다.

        Returns:
            tuple: (dominant_group, stats_dict)
                   e.g., ("20s_female", {"20s_female": 15, "40s_male": 8})
        """
        # 큐 안의 2D 리스트를 1D 리스트로 평탄화
        flat_list = [item for sublist in self.queue for item in sublist]

        if not flat_list:
            return None, {}  # (dominant_group, stats_dict)

        # Counter를 사용하여 각 항목의 개수를 셈
        stats = Counter(flat_list)
        
        # 가장 많이 등장한 항목(dominant group)을 찾음
        dominant_group = stats.most_common(1)[0][0]
        
        return dominant_group, dict(stats)