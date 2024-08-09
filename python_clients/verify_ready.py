import requests

def is_ready(ip, port):
# response = requests.get("http://[::1]:8000/v2/health/ready")
    response = requests.get(f"http://{ip}:{port}/v2/health/ready")
    if(response.status_code == 200):
        return True
    return False

print(is_ready("localhost", 8000))