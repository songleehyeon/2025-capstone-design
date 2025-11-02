import requests

def get_weather_context(api_key, city_name):
    """
    OpenWeatherMap API를 호출하여 현재 날씨를 기반으로
    컨텍스트 태그(e.g., "rainy_day")를 반환합니다.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }

    try:
        # 5초 타임아웃 설정
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()  # 200 OK가 아니면 예외 발생
        
        data = response.json()
        
        # API 응답에서 주 날씨 정보를 가져옴
        main_weather = data.get('weather', [{}])[0].get('main', 'Unknown')
        
        # 날씨 정보를 데모용 태그로 변환
        if main_weather == 'Rain':
            return 'rainy_day'
        elif main_weather == 'Snow':
            return 'snowy_day'
        elif main_weather == 'Clear':
            return 'sunny_day'
        elif main_weather == 'Clouds':
            return 'cloudy_day'
        else:
            return 'default_weather' # 'Mist', 'Haze' 등 기타 날씨

    except requests.exceptions.RequestException as e:
        print(f"Weather API Error: {e}")
        return "api_error"  # API 호출 실패 시