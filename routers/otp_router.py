import logging
from fastapi import APIRouter, HTTPException
from ml.isolation_forest import IsolationImplementaion
from sender.otp import OTP
from datetime import datetime, timedelta
from sender.dto import RequestOtpEntity, ValidateOtpEntity
from dbconf import conf

router = APIRouter()
otp = OTP()
@router.post("/request")
async def request_otp(request_otp_data: RequestOtpEntity):
    # isf = IsolationImplementaion()
    # req = {
    #     "pic_id": request_otp_data.phone_number,
    #     "purpose": "LOGIN",
    #     "latitude": "28.6139",#request_otp_data.geolocation.latitude,
    #     "longitude": "77.2090",#request_otp_data.geolocation.longitude,
    #     "device_name": "OnePlus 8T",#request_otp_data.device.device_name,
    #     "os_version": request_otp_data.device.os_version,
    #     "manufacturer": "OnePlus",#request_otp_data.device.manufacturer,
    #     "cpu_info": "Snapdragon 865",#request_otp_data.device.cpu_info,
    #     "platform": request_otp_data.device.platform,
    #     "ip": "203.123.123.123"#request_otp_data.ip
    # }
    # result = isf.predict(req)
    # if(result != "Anomali"):
    #     response = await otp.send_otp_request(phone=request_otp_data.phone_number)
    # else:
    #     logging.error("Anomali terdeteksi")
    #     raise HTTPException(status_code=401, detail="Anomali terdeteksi")
    response = await otp.send_otp_request(phone=request_otp_data.phone_number)
    current_time = datetime.now()
    expiry_time = current_time + timedelta(minutes=3)

    query = """
    INSERT INTO tb_otp_transaction (
        otp_id, merchant_id, provider_id, type, pic_id, otp, timestamp, purpose, latitude, 
        longitude, device_name, os_version, manufacturer, cpu_info, platform, ip, 
        is_active, expired_at, created_at
    ) VALUES (
        :otp_id, :merchant_id, :provider_id, :type, :pic_id, :otp, :timestamp, :purpose, 
        :latitude, :longitude, :device_name, :os_version, :manufacturer, :cpu_info, 
        :platform, :ip, :is_active, :expired_at, :created_at
    )
    """

    values = {
        "otp_id": response["data"]["id"],
        "merchant_id": 1,
        "provider_id": 1,
        "type": "REQUEST",  # asumsi ini adalah type
        "pic_id": request_otp_data.phone_number,
        "otp": response["data"]["otp"],
        "timestamp": current_time,
        "purpose": request_otp_data.purpose,
        "latitude": request_otp_data.geolocation.latitude,
        "longitude": request_otp_data.geolocation.longitude,
        "device_name": request_otp_data.device.device_name,
        "os_version": request_otp_data.device.os_version,
        "manufacturer": request_otp_data.device.manufacturer,
        "cpu_info": request_otp_data.device.cpu_info,
        "platform": request_otp_data.device.platform,
        "ip": request_otp_data.ip,
        "is_active": True,
        "expired_at": expiry_time,
        "created_at": current_time
    }
    await conf.database.execute(query, values)
    
    return response


@router.post("/verify")
async def verify_otp(validate_otp_data: ValidateOtpEntity):
    print(validate_otp_data)
    # Membuat request untuk verifikasi OTP ke service lain
    response = await otp.send_otp_verify(otp=validate_otp_data.otp, otp_id=validate_otp_data.otp_id)
    
    # Cek jika verifikasi sukses (asumsi respons memiliki key "data" dengan value "id" yang bernilai True jika sukses)
    if response["status"]==True:
        current_time = datetime.now()

        # Menyiapkan query untuk update database
        query = """
        UPDATE tb_otp_transaction
        SET is_active = :is_active, updated_at = :updated_at
        WHERE otp_id = :otp_id
        """

        values = {
            "otp_id": validate_otp_data.otp_id,
            "is_active": False,
            "updated_at": current_time
        }

        # Menjalankan kueri UPDATE ke database
        await conf.database.execute(query, values)
        
    return response
