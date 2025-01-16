import requests
from config import API_BASE_URL, PHOTO_ENDPOINT, ATTENDANCE_ENDPOINT

class APIClient:
    def __init__(self):
        self.base_url = API_BASE_URL

    def fetch_user_photos(self):
        """Получить все фотографии из базы данных через API."""
        response = requests.get(self.base_url + PHOTO_ENDPOINT)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code}")

    def record_attendance(self, user_id, check_in=True):
        """Записать вход или выход в таблицу attendance."""
        payload = {
            "user_id": user_id,
            "check_in": check_in
        }
        response = requests.post(self.base_url + ATTENDANCE_ENDPOINT, json=payload)
        if response.status_code == 201:
            print("Attendance recorded successfully")
        else:
            raise Exception(f"Failed to record attendance: {response.status_code}")
