# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 20:23:53 2020
这个节点作为服务端
@author: yzy86
"""
import socket
import get  # 自己写的
import threading
import os
import json
import numpy as np
import time

class ChatSever:

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addr = (get.get_ip(), 10000)
        self.users = {}
        self.res = []
        self.address = []
        self.tmp = []
        self.i = 0
        # self.index = 0

    def start_sever(self):
       	try:
        	self.sock.bind(self.addr)
        except Exception as e: print(e)
        self.sock.listen(5)
        print("服务器已开启，等待连接...")
        print("在空白处输入stop sever并回车，来关闭服务器")

        self.accept_cont()

    def accept_cont(self):
        while True:
            s, addr = self.sock.accept()
            self.users[addr] = s
            self.res.append(s)
            self.address.append(addr)
            number = len(self.users)
            print("用户{}连接成功！现在共有{}位用户".format(addr, number))

            threading.Thread(target=self.recv_send, args=(s, addr)).start()
            
    def recv_send(self, sock, addr):
        while True:
            try:  # 测试后发现，当用户率先选择退出时，这边就会报ConnectionResetError
                response = sock.recv(40960000).decode('utf8')
                mylist = json.loads(response)
                print('收到消息：', len(mylist))
                self.tmp.append(mylist)
                # print('address list is:',self.address)
                if len(self.tmp)>=4:
                    self.i += 1
                    print('calculating...')
                    addresult = np.array(self.tmp[0]) + np.array(self.tmp[1]) + np.array(self.tmp[2]) + np.array(self.tmp[3])
                    sp = addresult.shape
                    l=1
                    for s in sp:
                        l *= s
                    addresult = addresult.reshape(l)
                    print(addresult.shape)
                    addresult = addresult.tolist()
                    print('...done.')
                    if self.i != 4:
                        cut = int(len(addresult)/4)
                        print(cut)
                        print('slicing...')
                        result1 = addresult[:cut]
                        result2 = addresult[cut:2*cut]
                        result3 = addresult[2*cut:3*cut]
                        result4 = addresult[3*cut:]
                        json_result1 = json.dumps(result1)
                        json_result2 = json.dumps(result2)
                        json_result3 = json.dumps(result3)
                        json_result4 = json.dumps(result4)
                        result = [json_result1, json_result2, json_result3, json_result4]
                        self.tmp.pop(0)
                        self.tmp.pop(0)
                        self.tmp.pop(0)
                        self.tmp.pop(0)
                        print('sending...')
                        for i in range(len(self.res)):
                            self.res[i].send(result[i].encode('utf8'))
                            print('send the result...done', i)
                    else:
                        result1 = addresult
                        result2 = addresult
                        result3 = addresult
                        result4 = addresult
                        json_result1 = json.dumps(result1)
                        json_result2 = json.dumps(result2)
                        json_result3 = json.dumps(result3)
                        json_result4 = json.dumps(result4)
                        result = [json_result1, json_result2, json_result3, json_result4]
                        self.tmp.pop(0)
                        self.tmp.pop(0)
                        self.tmp.pop(0)
                        self.tmp.pop(0)
                        print('sending...')
                        for i in range(len(self.res)):
                            self.res[i].send(result[i].encode('utf8'))
                            # time.sleep(2)
                            print('send the result...done', i)
                # if len(self.tmp)>=2:
                #     self.i += 1
                #     print('calculating...')
                #     addresult = np.array(self.tmp[0]) + np.array(self.tmp[1])
                #     sp = addresult.shape
                #     l=1
                #     for s in sp:
                #         l *= s
                #     addresult = addresult.reshape(l)
                #     print(addresult.shape)
                #     addresult = addresult.tolist()
                #     print('...done.')
                #     if self.i != 4:
                #         cut = int(len(addresult)/4)
                #         print(cut)
                #         print('slicing...')
                #         result1 = addresult[:cut]
                #         result2 = addresult[cut:2*cut]
                #         json_result1 = json.dumps(result1)
                #         json_result2 = json.dumps(result2)
                #         result = [json_result1, json_result2]
                #         self.tmp.pop(0)
                #         self.tmp.pop(0)
                #         print('sending...')
                #         for i in range(len(self.res)):
                #             self.res[i].send(result[i].encode('utf8'))
                #             # time.sleep(2)
                #             print('send the result...done', i)
                #     else:
                #         result1 = addresult
                #         result2 = addresult
                #         json_result1 = json.dumps(result1)
                #         json_result2 = json.dumps(result2)
                #         result = [json_result1, json_result2]
                #         self.tmp.pop(0)
                #         self.tmp.pop(0)
                #         print('sending...')
                #         for i in range(len(self.res)):
                #             self.res[i].send(result[i].encode('utf8'))
                #             # time.sleep(2)
                #             print('send the result...done', i)
                        
                    
                    
                # msg = "{}用户{}发来消息：{}".format(get.get_time(), addr, response)
                # print(msg)
                # for client in self.users.values():
                #     client.send(msg.encode("utf8"))
            except ConnectionResetError:
                print("用户{}已经退出聊天！".format(addr))
                self.users.pop(addr)
                index = self.address.index(addr)
                self.address.remove(addr)
                del(self.res[index])
                break

    def close_sever(self):
        for client in self.users.values():
            client.close()
        self.sock.close()
        os._exit(0)


if __name__ == "__main__":
    sever = ChatSever()
    sever.start_sever()
    while True:
        cmd = input()
        if cmd == "stop sever":
            sever.close_sever()
        else:
            print("输入命令无效，请重新输入！")


