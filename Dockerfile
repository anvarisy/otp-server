# Gunakan gambar Python resmi
FROM python:3.11
ENV PYTHONUNBUFFERED True
# Set the timezone
RUN echo "Asia/Jakarta" > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Asia/Jakarta /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Buat direktori kerja
WORKDIR /usr/src/app

# Salin file kebutuhan dan instal paket yang diperlukan
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file sumber ke direktori kerja
COPY . .
ENV PORT 8080
EXPOSE 8080:8080
# Jalankan aplikasi
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080" ]