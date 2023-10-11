from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

class Geolocation(BaseModel):
    latitude: float
    longitude: float

    def to_dict(self):
        return {
            "latitude": self.latitude,
            "longitude": self.longitude
        }

class Device(BaseModel):
    device_name: str
    os_version: str
    manufacturer: str
    cpu_info: str
    platform: str

    def to_dict(self):
        return {
            "device_name": self.device_name,
            "os_version": self.os_version,
            "manufacturer": self.manufacturer,
            "cpu_info": self.cpu_info,
            "platform": self.platform
        }

class RequestOtpEntity(BaseModel):
    phone_number: str
    purpose: str
    ip: str
    package_name: str
    geolocation: Geolocation
    device: Device

    def to_dict(self):
        return {
            "phone_number": self.phone_number,
            "purpose": self.purpose,
            "ip": self.ip,
            "package_name": self.package_name,
            "geolocation": self.geolocation.to_dict(),
            "device": self.device.to_dict()
        }

class ValidateOtpEntity(BaseModel):
    otp: str
    otp_id: str
    ip: str
    package_name: str
    geolocation: Geolocation
    device: Device

    def to_dict(self):
        return {
            "otp": self.otp,
            "otp_id": self.otp_id,
            "ip": self.ip,
            "package_name": self.package_name,
            "geolocation": self.geolocation.to_dict(),
            "device": self.device.to_dict()
        }
