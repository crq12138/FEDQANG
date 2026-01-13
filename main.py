
import time
import sys
# print(sys.path)
# import blockchain
import threading
import key_Generate
# import MNIST_CNN_local
# import MNIST_training_DP
# import MNIST_training_pure
import MNIST_CNN
import CIFAR10_CNN
# import MNIST_Softmax
import p2p
# import decen_training
# from .blockchain import Blockchain

# _first_block = block.Block().first_block()
# print(_first_block)

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
    MNIST_CNN.run(0)
    # CIFAR10_CNN.run(0)
    # print("begin2")
