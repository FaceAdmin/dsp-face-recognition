from datetime import datetime
from api_client import APIClient
import re

def main():
    api = APIClient()    
    while True:
        code = input("Enter your entry code: ").strip()
        
        if code.lower() == "q":
            break
        
        if not re.fullmatch(r"\d{8}", code):
            print("[ERROR] Invalid Input")
            continue
        
        try:
            user_data = api.get_user_by_code(code)
            user_id = user_data.get("user_id")
            if not user_id:
                print("[ERROR] This code is not associated with any user.")
                continue
            
            api.record_attendance(user_id)
            if "fname" in user_data and "lname" in user_data:
                full_name = f"{user_data['fname']} {user_data['lname']}"
            else:
                user_details = api.get_user(user_id)
                full_name = f"{user_details.get('fname', 'Unknown')} {user_details.get('lname', '')}".strip()
            
            check_in_date = datetime.now().strftime("%d.%m.%Y")
            print(f"Check in: {check_in_date}\nUser: {full_name}")
            print("[INFO] Attendance recorded successfully.")
        except Exception as e:
            print(f"[ERROR] {e}")
        
        print("")

if __name__ == "__main__":
    main()
