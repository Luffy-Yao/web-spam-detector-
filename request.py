#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 13:08:17 2020

@author: linxing
"""

import requests
import json

data = {'host':"com.safetyserve.course", 
        'url':"/FineSource/elearning/courses/titles/Lib_StoryLineR3CT/course_hi_lk/story_content/slides/5kFJL7ZUa08.swf"}
url = 'http://127.0.0.1:1234' 
data=json.dumps(data)
# send requests
send_request=requests.post(url,data)
print(send_request.json())