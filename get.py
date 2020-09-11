# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:59:22 2020

@author: yzy86
"""


import socket
import datetime


def get_ip():
    """用来搞到IP"""
    host = socket.gethostname()
    ip = socket.gethostbyname(host)
    return ip


def get_time():
    """得到发送时间"""
    now = datetime.datetime.now()
    send_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return send_time
