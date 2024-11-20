import os
import time
import urllib.request


# change these parameter
dirname = "dataset/captchas"
number = 1000
start = 10001
urls = ["https://ems1.sbu.ac.ir/Forms/AuthenticateUser/captcha.aspx",
    "https://golestan.iust.ac.ir/Forms/AuthenticateUser/captcha.aspx",
    "https://golestan.ikiu.ac.ir/Forms/AuthenticateUser/captcha.aspx",
    "https://osreg.pnu.ac.ir/Forms/AuthenticateUser/captcha.aspx",
    "https://education.cfu.ac.ir/Forms/AuthenticateUser/captcha.aspx",
    "https://golestan.ui.ac.ir/Forms/AuthenticateUser/captcha.aspx",
    "https://ems.atu.ac.ir/Forms/AuthenticateUser/captcha.aspx",
    "https://edu.qom.ac.ir/Forms/AuthenticateUser/captcha.aspx",
    "https://golestan.uok.ac.ir/Forms/AuthenticateUser/captcha.aspx",
    "https://golestan.yazd.ac.ir/Forms/AuthenticateUser/captcha.aspx",
    "https://golestan.uok.ac.ir/Forms/AuthenticateUser/captcha.aspx"]


if not os.path.exists(dirname):
    os.makedirs(dirname)

os.chdir(dirname)

counter = 0
while counter < number:

    filename = f"{start:05}" + ".gif"

    url = urls[counter % len(urls)]
    
    try:
        out = urllib.request.urlretrieve(url, filename)
        print(out, url)

        counter += 1
        start += 1
    except:
        print("An exception occurred") 

    time.sleep(1)