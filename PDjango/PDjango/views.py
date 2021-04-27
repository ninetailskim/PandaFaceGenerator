from django.http import HttpResponse
import json
import os
from . import pandaFace
import time
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def hello(request):
    return HttpResponse("Hello World!")

def meme(request):

    imgstr = request.GET.get('image')
    text = request.GET.get('text')
    print(text)
    load_dict = None
    with open("Q:\\LAB\\Django\\wechaty-getting-started\\sender.json", "r",encoding='utf-8') as fp:
        load_dict = json.load(fp)
    if text in load_dict:
        text = load_dict[text]
    else:
        text = None

    print(imgstr)
    res = None
    try:
        res = pandaFace.PF.finalCompose(imgstr, text)
        # time.sleep(1)
    except Exception as e:
        print(e)

    print("-----------", res)
    if res is None:
        content = {'name': "I can't found face in your picture", "status":"False"}
    else:
        content = {'name':os.path.abspath(res), "status":"True"}
    content = json.dumps(content)
    response = HttpResponse(content=content, content_type='application/json')
    response.status_code = 200
    return response

    
