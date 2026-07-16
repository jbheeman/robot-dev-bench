import urllib.request, re
url = "https://www.bing.com/images/search?q=unitree+g1+humanoid+robot"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8')
urls = re.findall(r'murl&quot;:&quot;(http[^&]+)&quot;', html)
if urls:
    import requests
    # some urls might be blocked, try first 5
    for img_url in urls[:5]:
        print("Trying:", img_url)
        try:
            r = requests.get(img_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if r.status_code == 200:
                with open('unitree_g1.jpg', 'wb') as f:
                    f.write(r.content)
                print("Success")
                break
        except:
            pass
