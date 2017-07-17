import grequests
import numpy as np
import _pickle as cPickle
from _pickle import dumps, loads
import base64
from time import sleep
import json

img = np.zeros([100,100,3],dtype=np.uint8)
img.fill(255) # or img[:] = 255
img=dumps(img)
#img= base64.b64encode(img)
#print (img)


#print(r.status_code, r.reason,r.text)


count=0
busy=False

def manageAnalResponse(response, **kwargs):
    global busy
    print("got reponse")
    print (json.loads(response.text)[0])
    busy=False

while True:
    count+=1
    print(count)
    if not busy:
        print("not busy")
        busy=True
        r = ([grequests.post("http://127.0.0.1:23948/analysiseFrame", data=img,headers={'Content-Type': 'application/octet-stream'},hooks={'response':manageAnalResponse})])
        grequests.map(r)
        #print (r)
    
    sleep(1)

