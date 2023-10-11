import requests

class OTP:
    def __init__(self):
        pass
    
    async def send_otp_request(self, phone: str):
        base_url = "https://api.fazpas.com/v1"
        request_url = "/otp/generate"

        headers = {
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZGVudGlmaWVyIjo0fQ.WEV3bCizw9U_hxRC6DxHOzZthuJXRE8ziI3b6bHUpEI"
        }

        body = {
            "phone": phone,
            "gateway_key": "8521255b-1291-4b59-a1a4-794cb1645deb",
            "external_id": ""
        }
        response = requests.post(f"{base_url}{request_url}", json=body, headers=headers)
        return response.json()
           
    
    async def send_otp_verify(self, otp: str, otp_id:str):
        base_url = "https://api.fazpas.com/v1"
        request_url = "/otp/verify"

        headers = {
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZGVudGlmaWVyIjo0fQ.WEV3bCizw9U_hxRC6DxHOzZthuJXRE8ziI3b6bHUpEI"
        }

        body = {
           "otp": otp,
            "otp_id": otp_id,
        }

        
        response = requests.post(f"{base_url}{request_url}", json=body, headers=headers)
        return response.json()
            