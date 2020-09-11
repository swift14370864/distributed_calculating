# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:30:41 2020
client
@author: yzy86
"""

import socket
import threading
hostport = ('127.0.0.1',9999)
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect(hostport)
true = True
def Receve(s):
    global true
    while true:
        data = s.recv(1024).decode('utf8')
        if data == 'quit':
            true = False
        print('recevie news:\033[5;37;46m%s\033[0m' % data )
thrd=threading.Thread(target=Receve,args=(s,))
thrd.start()
while true:
    user_input = input('>>>')
    s.send(user_input.encode('utf8'))
    if user_input == 'quit':
        true = False
s.close()