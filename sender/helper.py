from datetime import datetime, timedelta
import random
import uuid

# Data awal
PHONE_NUMBERS = ["085811751000", "085195310101"]
PURPOSES = ["LOGIN", "TRANSACTION"]
IPS = ["192.168.1.34", "192.168.1.36", "10.5.248.250"]
DEVICE_NAMES = ["SM-A235F", "Infinix X670","OnePlus 8T", "Xiaomi Redmi Note 10"]
MANUFACTURERS = ["samsung", "Infinix", "OnePlus", "Xiaomi"]
CPU_INFOS = ["Qualcomm Technologies, Inc KHAJE", "MT6781V/CDZAMB-H", "Snapdragon 865", "Mediatek Helio G95"]
LATITUDE_BASE = -6.6502314
LONGITUDE_BASE = 106.7560309

def generate_seed_data(records: int = 10)->list:
    data = []
    current_time = datetime.now()
    
    for _ in range(records):
        # Variasi waktu antara record
        current_time -= timedelta(days=random.choice([1, 2, 3, 4, 5]))
        expiry_time = current_time + timedelta(minutes=3)
        updated_at = current_time + timedelta(seconds=random.randint(10, 20))

        pic_id = random.choice(PHONE_NUMBERS)
        purpose = random.choice(PURPOSES)
        manufacture = random.choice(MANUFACTURERS)
        cpu_info = random.choice(CPU_INFOS)
        device_name = random.choice(DEVICE_NAMES)
        ip = random.choice(IPS)

        # Geser geolocation sekitar 50-100 meter
        latitude = LATITUDE_BASE + (random.randint(-100, 100) * 0.000009)
        longitude = LONGITUDE_BASE + (random.randint(-100, 100) * 0.000009)

        record = {
            "otp_id": str(uuid.uuid4()),
            "merchant_id": 1,
            "provider_id": 1,
            "type": "REQUEST",
            "pic_id": pic_id,
            "otp": str(random.randint(1000, 9999)),
            "timestamp": current_time,
            "purpose": purpose,
            "latitude": latitude,
            "longitude": longitude,
            "device_name": device_name,
            "os_version": "13",
            "manufacturer": manufacture,
            "cpu_info": cpu_info,
            "platform": "ANDROID",
            "ip": ip,
            "is_active": False,
            "expired_at": expiry_time,
            "created_at": current_time,
            "updated_at": updated_at
        }

        data.append(record)

    return data


INDIAN_IPS = ["203.123.123.123", "203.125.125.125"]
INDIAN_DEVICE_NAMES = ["OnePlus 8T", "Xiaomi Redmi Note 10"]
INDIAN_MANUFACTURERS = ["OnePlus", "Xiaomi"]
INDIAN_CPU_INFOS = ["Snapdragon 865", "Mediatek Helio G95"]
LATITUDE_INDIAN_BASE = 28.6139
LONGITUDE_INDIAN_BASE = 77.2090

def generate_augmented_data(original_data: list)->list:
    augmented_data = []
    for record in original_data:
        # Mengubah data original dengan data India
        record["ip"] = random.choice(INDIAN_IPS)
        record["device_name"] = random.choice(INDIAN_DEVICE_NAMES)
        record["manufacturer"] = random.choice(INDIAN_MANUFACTURERS)
        record["cpu_info"] = random.choice(INDIAN_CPU_INFOS)

        # Geser geolocation sekitar 50-100 meter dari Delhi
        record["latitude"] = LATITUDE_INDIAN_BASE + (random.randint(-100, 100) * 0.000009)
        record["longitude"] = LONGITUDE_INDIAN_BASE + (random.randint(-100, 100) * 0.000009)

        augmented_data.append(record)
    
    return augmented_data