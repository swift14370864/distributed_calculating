# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:18:10 2020

@author: yzy86
"""


import socket
import threading
import os
import get

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# addr = # 服务器端的IP和端口
addr = (get.get_ip(), 10000)#。只有自己一台电脑做测试时，可以直接用左边的
s.connect(addr)


def recv_msg():  #
    print("连接成功！现在可以接收消息！\n")
    while True:
        try:  # 测试发现，当服务器率先关闭时，这边也会报ConnectionResetError
            response = s.recv(1024)
            print(response.decode("utf8"))
        except ConnectionResetError:
            print("服务器关闭，聊天已结束！")
            s.close()
            break
    os._exit(0)


def send_msg():
    print("连接成功！现在可以发送消息！\n")
    print("输入消息按回车来发送")
    print("输入esc来退出聊天")
    while True:
        msg = input(">>>:").strip()
        if not msg: continue
        if msg == "esc":
            print("你退出了聊天")
            s.close()
            break
        s.send(msg.encode("utf8"))
    os._exit(0)


threads = [threading.Thread(target=recv_msg), threading.Thread(target=send_msg)]
for t in threads:
    t.start()
