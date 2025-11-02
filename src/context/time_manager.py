import datetime

def get_time_context():
    """
    현재 시간을 기준으로 컨텍스트 태그(e.g., "morning_rush")를 반환합니다.
    """
    now = datetime.datetime.now()
    hour = now.hour

    if 7 <= hour < 10:
        return "morning_rush"
    elif 11 <= hour < 14:
        return "lunch_time"
    elif 18 <= hour < 20:
        return "evening_rush"
    elif 20 <= hour:
        return "night_time"
    else:
        return "day_time"