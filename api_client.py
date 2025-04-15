import datetime
import requests
from config import API_BASE_URL, PHOTO_ENDPOINT, ATTENDANCE_ENDPOINT, USER_ENDPOINT

class APIClient:
    def __init__(self):
        self.base_url = API_BASE_URL
        self.session = requests.Session()

    def fetch_user_photos(self):
        url = self.base_url + PHOTO_ENDPOINT
        response = self.session.get(url)
        if response.ok:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code}")
        
    def fetch_aggregated_encodings(self):
        url = f"{self.base_url}/photos/get-encodings/"
        response = self.session.get(url)
        if response.ok:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code}, {response.text}")

    def record_attendance(self, user_id: int):
        user_data = self.get_user(user_id)
        first_name = user_data.get("first_name", "")
        last_name = user_data.get("last_name", "")
        email = user_data.get("email", "")

        current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        url = f"{self.base_url}{ATTENDANCE_ENDPOINT}?user_id={user_id}"
        resp = self.session.get(url)
        if not resp.ok:
            raise Exception(f"Failed to fetch attendance records: {resp.status_code}")

        records = resp.json()
        if records and records[-1]["check_out"] is None:
            attendance_id = records[-1]["attendance_id"]
            payload = {"check_out": current_time}
            patch_url = f"{self.base_url}{ATTENDANCE_ENDPOINT}{attendance_id}/"
            resp_patch = self.session.patch(patch_url, json=payload)
            if resp_patch.ok:
                print("[INFO] Check-out recorded")
                log_message = f"{first_name} {last_name} ({email}) checked out."
                self.session.post(f"{self.base_url}/logs/", json={"user": user_id, "action": log_message})
            else:
                raise Exception(f"Failed to record check-out: {resp_patch.status_code}")
        else:
            payload = {"user": user_id, "check_in": current_time}
            post_url = self.base_url + ATTENDANCE_ENDPOINT
            resp_post = self.session.post(post_url, json=payload)
            if resp_post.status_code == 201:
                print("[INFO] Check-in recorded")
                log_message = f"{first_name} {last_name} ({email}) checked in."
                self.session.post(f"{self.base_url}/logs/", json={"user": user_id, "action": log_message})
            else:
                raise Exception(f"Failed to record check-in: {resp_post.status_code}")

    def get_user(self, user_id: int):
        url = f"{self.base_url}{USER_ENDPOINT}{user_id}/"
        resp = self.session.get(url)
        if resp.ok:
            return resp.json()
        else:
            raise Exception(f"Failed to fetch user: {resp.status_code}")
        
    def verify_otp(self, email: str, otp_code: str):
            url = f"{self.base_url}/users/verify-otp/"
            data = {"email": email, "otp_code": otp_code}
            resp = self.session.post(url, json=data)
            if resp.status_code == 200:
                return resp.json()
            else:
                raise Exception(f"OTP verification failed: {resp.text}")