import json

def load_ad_database(json_path):
    """
    지정된 경로의 광고 DB JSON 파일을 로드합니다.

    Args:
        json_path (str): 'config/ad_db.json' 파일 경로

    Returns:
        dict: 광고 정보 딕셔너리
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Ad database file not found at {json_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_path}")
        return {}