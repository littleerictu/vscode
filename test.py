import requests
import certifi

def check_tsinghua_mirror():
    url = "https://pypi.tuna.tsinghua.edu.cn/simple"
    try:
        response = requests.get(url, verify=certifi.where())
        if response.status_code == 200:
            print("清华大学的 PyPI 镜像可用")
        else:
            print("清华大学的 PyPI 镜像不可用，状态码:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("请求失败:", e)

print("Hello, Python for MAC!")
check_tsinghua_mirror()