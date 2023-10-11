from geopy.distance import geodesic
import pandas as pd

class Preprocess:
    def __init__(self):
        pass
        
    def start(self, data: list):
        df = pd.DataFrame(data)
        # 1. Mengurutkan data
        df.sort_values(by=['pic_id', 'timestamp'], ascending=[True, True], inplace=True)
        # 2. Membuat kolom is_change_device
        df['is_change_device'] = (df['manufacturer'] != df['manufacturer'].shift()) | (df['device_name'] != df['device_name'].shift()) | (df['cpu_info'] != df['cpu_info'].shift())
        df.loc[df.groupby('pic_id').head(1).index, 'is_change_device'] = False
        # 3. Membuat kolom response_time
        df['response_time'] = (df['updated_at'] - df['created_at']).dt.total_seconds()
        df['latitude_shift'] = df['latitude'].shift()
        df['longitude_shift'] = df['longitude'].shift()
        df['distance'] = df.apply(self.calculate_distance, axis=1)
        df.drop(columns=['latitude_shift', 'longitude_shift'], inplace=True)

        # 5. Membuat kolom is_ip_change
        df['is_ip_change'] = df['ip'] != df['ip'].shift()
        df.loc[df.groupby('pic_id').head(1).index, 'is_ip_change'] = False

        # 6. Membuat kolom validation_status
        df['validation_status'] = ~df['is_active']
        df.to_csv('modified_data.csv', index=False)
        for pic_id, group in df.groupby('pic_id'):
            group.to_csv(f'data_{pic_id}.csv', index=False)
        print(df)

# 4. Membuat kolom distance
    def calculate_distance(self, row):
        prev_lat, prev_long = row['latitude_shift'], row['longitude_shift']
        curr_lat, curr_long = row['latitude'], row['longitude']
        
        if pd.isnull(prev_lat) or pd.isnull(prev_long):
            return 0
        return geodesic((prev_lat, prev_long), (curr_lat, curr_long)).meters

