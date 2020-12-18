import json
import urllib.request
import os
import socket
import requests

socket.setdefaulttimeout(15)
data_root = './'
base_url = 'https://yande.re/post.json?tags= width:1920 height:1080&page='
os.chdir(data_root)

k = 0
for i in range(1, 208):
    s = base_url + str(i)
    try:
        json_response = json.loads(requests.get(s).text)
        for j in range(len(json_response)):
            st = json_response[j]['file_url']
            urllib.request.urlretrieve(st, "yande_" + str(k) + '.' + st.split('.')[-1])
            if k % 10 == 0:
                print(str(k))
            k+= 1
    except:
        print("error: " + str(k))

