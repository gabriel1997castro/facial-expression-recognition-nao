# Echo client program
import socket

HOST = '127.0.0.1'    # The remote host
PORT = 4321              # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
hello = 'Hello, world'
s.sendall(hello.encode('utf-8'))
data = s.recv(1024)
s.close()
print('Received ', repr(data))