import urllib.request
import uuid

boundary = uuid.uuid4().hex

with open('/home/andrew/Downloads/episode_000000.parquet', 'rb') as f:
    file_data = f.read()
    
data = b'--' + boundary.encode() + b'\r\nContent-Disposition: form-data; name="task"\r\n\r\nwalking\r\n--' + boundary.encode() + b'\r\nContent-Disposition: form-data; name="file"; filename="episode_000000.parquet"\r\nContent-Type: application/octet-stream\r\n\r\n' + file_data + b'\r\n--' + boundary.encode() + b'--\r\n'

req = urllib.request.Request('http://127.0.0.1:8000/api/upload', data=data, headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})

try:
    resp = urllib.request.urlopen(req)
    print("STATUS:", resp.status)
    print("BODY:", resp.read().decode()[:500])
except Exception as e:
    print("ERROR:", e)
    if hasattr(e, 'read'):
        print("BODY:", e.read().decode())
