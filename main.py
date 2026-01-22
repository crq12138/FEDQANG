
import time
import sys
# print(sys.path)
# import blockchain
# import threading
import MNIST_CNN
import CIFAR10_CNN
import MedMNIST_CNN
import p2p


if __name__ == '__main__':
    # p2p-grpc initializaiton
    # file = open("ipport.txt", 'w').close()
    ipport=sys.argv[1]
    # ipport="127.0.0.1"
    port=sys.argv[2]
    # port = "50052"
    p2p.set_address(ipport, port)
    Node = p2p.Node()
    Node.grpcNetworkStart()
    time.sleep(15)
    # print("begin2")
    # MNIST_training_DP.run(0)
    # MNIST_Softmax.run(0)
    # MNIST_CNN.run(0)
    MedMNIST_CNN.run(0)
    # CIFAR10_CNN.run(0)
    # print("begin2")
