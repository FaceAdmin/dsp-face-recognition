import datetime
import requests
from config import API_BASE_URL, PHOTO_ENDPOINT, ATTENDANCE_ENDPOINT, USER_ENDPOINT, ENTRY_CODE_ENDPOINT

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
        user_data = self.get_user(user_id)
        fname = user_data.get("fname", "")
        lname = user_data.get("lname", "")
        email = user_data.get("email", "")

        current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        url = f"{self.base_url}{ATTENDANCE_ENDPOINT}?user_id={user_id}"
        resp = requests.get(url)
        if not resp.ok:
            raise Exception(f"Failed to fetch attendance records: {resp.status_code}")

        records = resp.json()
        if records and records[-1]["check_out"] is None:
            attendance_id = records[-1]["attendance_id"]
            payload = {"check_out": current_time}
            patch_url = f"{self.base_url}{ATTENDANCE_ENDPOINT}{attendance_id}/"
            resp_patch = requests.patch(patch_url, json=payload)
            if resp_patch.ok:
                print("[INFO] Check-out recorded")
                log_message = f"{fname} {lname} ({email}) checked out."
                requests.post(f"{self.base_url}/logs/", json={"user": user_id, "action": log_message})
            else:
                raise Exception(f"Failed to record check-out: {resp_patch.status_code}")
        else:
            payload = {"user": user_id, "check_in": current_time}
            post_url = self.base_url + ATTENDANCE_ENDPOINT
            resp_post = requests.post(post_url, json=payload)
            if resp_post.status_code == 201:
                print("[INFO] Check-in recorded")
                log_message = f"{fname} {lname} ({email}) checked in."
                requests.post(f"{self.base_url}/logs/", json={"user": user_id, "action": log_message})
            else:
                raise Exception(f"Failed to record check-in: {resp_post.status_code}")

    def get_user_by_code(self, code: str):
        url = f"{self.base_url}{ENTRY_CODE_ENDPOINT}?code={code}"
        resp = requests.get(url)
        if resp.ok:
            data = resp.json()
            if data:
                user = data.get("user")
                if not user:
                    raise Exception("Entry code not linked to any user_id")
                
                user_details = self.get_user(user)
                return user_details
            else:
                raise Exception("Entry code not found")
        else:
            raise Exception(f"Failed to fetch entry code: {resp.status_code}")

    def get_user(self, user_id: int):
        url = f"{self.base_url}{USER_ENDPOINT}{user_id}/"
        resp = requests.get(url)
        if resp.ok:
            return resp.json()
        else:
            raise Exception(f"Failed to fetch user: {resp.status_code}")
