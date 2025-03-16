import datetime
import requests
from config import API_BASE_URL, PHOTO_ENDPOINT, ATTENDANCE_ENDPOINT

class APIClient:
    def __init__(self):
        self.base_url = API_BASE_URL

    def fetch_user_photos(self):
        url = self.base_url + PHOTO_ENDPOINT
        response = requests.get(url)
        if response.ok:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code}")

    def record_attendance(self, user_id: int):
        current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

        url = f"{self.base_url}{ATTENDANCE_ENDPOINT}?user_id={user_id}"
        resp = requests.get(url)
        if not resp.ok:
            raise Exception(f"Failed to fetch attendance records: {resp.status_code}")

        records = resp.json()
        if records and records[-1]["check_out"] is None:
            attendance_id = records[-1]["attendance_id"]
            payload = {
                "check_out": current_time
            }
            patch_url = f"{self.base_url}{ATTENDANCE_ENDPOINT}{attendance_id}/"
            resp_patch = requests.patch(patch_url, json=payload)
            if resp_patch.ok:
                print(f"[INFO] Check-out recorded for user {user_id}.")
            else:
                raise Exception(f"Failed to record check-out: {resp_patch.status_code}")
        else:
            payload = {
                "user": user_id,
                "check_in": current_time
            }
            post_url = self.base_url + ATTENDANCE_ENDPOINT
            resp_post = requests.post(post_url, json=payload)
            if resp_post.status_code == 201:
                print(f"[INFO] Check-in recorded for user {user_id}.")
            else:
                raise Exception(f"Failed to record check-in: {resp_post.status_code}")
