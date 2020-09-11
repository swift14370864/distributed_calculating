# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:30:47 2020
server
@author: yzy86
"""

import socket
import threading                                                # 导入多线程模块
print("Waitting to be connected......")
HostPort = ('127.0.0.1',9999)
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)            # 创建socket实例
s.bind(HostPort)
s.listen(1)
conn,addr = s.accept()
true=True
addr = str(addr)
print('Connecting by : %s ' %addr )
def Receve(conn):                                               # 将接收定义成一个函数
    global true                                                 # 声明全局变量，当接收到的消息为quit时，则触发全局变量 true = False，则会将socket关闭
    while true:
        data = conn.recv(1024).decode('utf8')          
        if data == 'quit':
            true=False
        print("you have receve: "+data+" from"+addr)            # 当接收的值为'quit'时，退出接收线程，否则，循环接收并打印
thrd=threading.Thread(target=Receve,args=(conn,))               # 线程实例化，target为方法，args为方法的参数 
thrd.start()                                                    # 启动线程
while true:
    user_input = input('>>>')
    conn.send(user_input.encode('utf8'))                        # 循环发送消息
    if user_input == 'quit':                                    # 当发送为‘quit’时，关闭socket
        true = False
    #conn.close()
s.close()
