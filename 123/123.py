import requests

def send_request(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  
        return response.text

    except requests.exceptions.ConnectionError:
        print(" اتصال به سرور برقرار نشد. لطفاً اینترنت خود را بررسی کنید.")
    
    except requests.exceptions.Timeout:
        print(" درخواست بیش از حد طول کشید. دوباره تلاش کنید.")
    
    except requests.exceptions.HTTPError as e:
        print(f" سرور خطا برگرداند: {e.response.status_code}")
    
    except Exception as e:
        print(f" یک خطای ناشناخته رخ داد: {str(e)}")


# مثال اجرا
send_request("https://example.com/api/data")
