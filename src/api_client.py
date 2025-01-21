import datetime
import requests
from config import API_BASE_URL, PHOTO_ENDPOINT, ATTENDANCE_ENDPOINT

class APIClient:
    def __init__(self):
        self.base_url = API_BASE_URL

    def fetch_user_photos(self):
        response = requests.get(self.base_url + PHOTO_ENDPOINT)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code}")
        
    def calculate_duration(self, check_in, check_out):
        check_in_time = datetime.datetime.fromisoformat(check_in.replace('Z', '+00:00'))
        check_out_time = datetime.datetime.fromisoformat(check_out.replace('Z', '+00:00'))
        duration = check_out_time - check_in_time
        return str(duration)

    def record_attendance(self, user_id, check_in_time=None):
        response = requests.get(f"{self.base_url}{ATTENDANCE_ENDPOINT}?user_id={user_id}")
        if response.status_code == 200:
            attendances = response.json()
            if attendances and attendances[-1]["check_out"] is None:
                payload = {
                    "check_out": check_in_time,
                    "duration": self.calculate_duration(attendances[-1]["check_in"], check_in_time),
                }
                attendance_id = attendances[-1]["attendance_id"]
                response = requests.patch(f"{self.base_url}{ATTENDANCE_ENDPOINT}{attendance_id}/", json=payload)
                if response.status_code == 200:
                    print("Check-out recorded successfully.")
                else:
                    raise Exception(f"Failed to record check-out: {response.status_code}")
            else:
                payload = {
                    "user": user_id,
                    "check_in": check_in_time,
                }
                response = requests.post(self.base_url + ATTENDANCE_ENDPOINT, json=payload)
                if response.status_code == 201:
                    print("Check-in recorded successfully.")
                else:
                    raise Exception(f"Failed to record check-in: {response.status_code}")
        else:
            raise Exception(f"Failed to fetch attendance records: {response.status_code}")

