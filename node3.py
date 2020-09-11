# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:25:29 2020
node3
@author: yzy86
"""

import socket
hostport = ('127.0.0.1',9999)
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect(hostport)
 
while 1:
    user_input = input('>>> ').strip()
    s.send(user_input.encode('utf-8'))
    if len(user_input) == 0:
        continue
    if user_input == 'quit':
        s.close()
        break
    server_reply = s.recv(1024).decode()
    print (server_reply)