# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:25:45 2020
node4
@author: yzy86
"""


from socket import *


client = socket()
ip_port = ("127.0.0.1", 8080)
client.connect(ip_port)
while 1:
    inp = input(">>>:").strip()
    if not inp: continue
    client.send(inp.encode("utf-8"))
    from_server_msg = client.recv(1024)
    print("来自服务端的消息:", from_server_msg)

client.close()