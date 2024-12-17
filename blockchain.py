# coding:utf-8
import grpc
# from ecdsa import SigningKey, NIST384p, VerifyingKey
# from networkx import generate_adjlist
# from pyparsing import Opt
import grpc_pb2
import grpc_pb2_grpc
import time
# import re
import json
import hashlib
# import threading
import p2p
# import numpy as np
import pickle
from Hashring import HashRing
import threading
# from ecdsa import SigningKey, VerifyingKey, NIST384p
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
# _compiNum = re.compile("^\d+$")  # 判斷全數字用
# _compiW = re.compile("^\w{64}")

max_message_length = 100 * 1024 * 1024  # 设置为 100 MB，可根据需要调整
options = [
    ('grpc.max_send_message_length', max_message_length),
    ('grpc.max_receive_message_length', max_message_length),
]

def generate_rsa_keys():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    return private_key, public_key

# TODO
with open('svkey.json', 'r', encoding='utf8')as fp:
    svkey = json.load(fp)

# def encrypt_block(block, public_key):
#     # block_bytes = pickle.dumps(block)
#     print(len(block))
#     encrypted_block = public_key.encrypt(
#         block,
#         padding.OAEP(
#             mgf=padding.MGF1(algorithm=hashes.SHA256()),
#             algorithm=hashes.SHA256(),
#             label=None
#         )
#     )
#     return encrypted_block

# def decrypt_block(encrypted_block, private_key):
#     block_bytes = private_key.decrypt(
#         encrypted_block,
#         padding.OAEP(
#             mgf=padding.MGF1(algorithm=hashes.SHA256()),
#             algorithm=hashes.SHA256(),
#             label=None
#         )
#     )
#     # block = pickle.loads(block_bytes)
#     return block_bytes

def encrypt_data_block(data_block, key):
    # 创建加密器（AES CBC模式）
    cipher = AES.new(key, AES.MODE_CBC)
    # 对数据块进行填充并加密
    ciphertext = cipher.encrypt(pad(data_block, AES.block_size))
    return cipher.iv, ciphertext

# 定义解密函数
def decrypt_data_block(iv, ciphertext, key):
    # 创建解密器（使用相同的密钥和初始化向量）
    cipher = AES.new(key, AES.MODE_CBC, iv)
    # 解密并移除填充
    data_block = unpad(cipher.decrypt(ciphertext), AES.block_size)
    return data_block

def signing(privatekey, data):
    # return base64.b64encode(str((privatekey.sign(data, ''))[0]).encode())
    return privatekey.sign(bytes(data))


def verifying(publickey, data, sign):
    # return publickey.verify(data, (int(base64.b64decode(sign)),))
    assert publickey.verify(sign, bytes(data))
    return True


def hash_block(block: grpc_pb2.Block) -> str:
    hash_str = block.SerializeToString()
    s = hashlib.sha256()  # Get the hash algorithm.
    s.update(hash_str)  # Hash the data.
    hash = s.hexdigest()  # Get he hash value.
    return hash


global pre_prepare_receive
global final_block_receive
block_list = list()
key_list = list()
# final_block = None
# grad_list = list()

class Blockchain:
    def __init__(self):
        self.node_id = p2p.PORT
        self.ipport = p2p.SELF_IP_PORT
        self.nodes = set(p2p.Node.get_nodes_list())
        self.chain = []
        self.committee = set()
        self.committee_size = 3  # 可以根据需要调整
        self.hash_ring = HashRing()
        self.initialize_hash_ring()
        self.lastBlock = None
        self.final_block = None
        # Genesis block
        block = self.create_block(None)
        self.add_block(block)
        # self.lock = threading.Lock()

    def initialize_hash_ring(self):
        nodes = set(p2p.Node.get_nodes_list())
        # print(nodes)
        sorted_nodes = sorted(nodes)
        print(sorted_nodes)
        # 假设我们有一个全局的客户端列表，可以遍历并添加到哈希环
        for node_address in sorted_nodes:
            # client = self.get_client_by_address(node_address)
            # if client:
            #     weight = client.get_Hash_stake()
            #     self.hash_ring.add_node(node_address, weight)
            weight = 70
            self.hash_ring.add_node(node_address, weight)
    
    # def get_client_by_address(self, address):
    #     # 根据地址获取对应的 Client 实例
    #     # 这里假设有一个全局的客户端列表或映射
    #     # 需要根据实际情况实现
    #     for client in global_clients_list:
    #         if client.port == address.split(':')[1]:
    #             return client
    #     return None

    def elect_committee(self):
        # print(self.lastBlock)
        if not self.lastBlock:
            seed = "genesis"
        else:
            seed = hash_block(self.lastBlock)
        print("seed is ", seed)
        self.committee = self.hash_ring.get_nodes_for_committee(seed, self.committee_size)
        print("委员会成员选举完成:", self.committee)


    def add_block(self, block):
        self.chain.append(block)
        self.lastBlock = block
        self.elect_committee()
        
        global key_list, block_list, final_block_receive
        # key_list.clear()
        # block_list.clear()
        final_block_receive = None
        self.final_block = None

    def create_block(self,tensor) -> grpc_pb2.Block:
        if self.lastBlock is None:
            block = grpc_pb2.Block(
                height=1,
                timestamp=0,
                previoushash=b'',
                txshash=[],
                krumgrad=b''
            )
            # print("tensor is :", tensor)
        else:
            block = grpc_pb2.Block(
                height=self.lastBlock.height + 1,
                # timestamp=int(time.time()),
                previoushash=hash_block(self.lastBlock),
                txshash=[],
                krumgrad=tensor
            )
        # print("tensor is :", tensor)
        return block


    def block_hash(self, Block):
        return Block.block_hash

    def block_height(self, block):
        return block.height

    def block_tx(self, block):
        return block.tx


    def consensus_process(self, krum_grad_bytes):
        # global final_block
        global key_list, block_list
        # global final_block_receive
        # sk = SigningKey.from_string(bytes.fromhex(svkey[self.node_id][0]), curve=NIST384p)

        # 1. 创建备选区块
        candidate_block = self.create_block(krum_grad_bytes)
        block_bytes = pickle.dumps(candidate_block)

        # 2. 生成密钥对
        key = get_random_bytes(16)

        # 3. 对区块进行加密
        iv, encrypted_block = encrypt_data_block(block_bytes, key)
        # print(type(iv))
        # print(type(encrypted_block))
        t = threading.Thread(target=self.Block_send(encrypted_block))
        t.start()
        t.join()
        # self_node = set()
        # self_node.add(p2p.SELF_IP_PORT)
        # nodes = self.committee - self_node

        # # 加密区块作为发送的消息
        # request = grpc_pb2.Blockmessage()
        # print(type(self.node_id))
        # request.Data.node_id = self.node_id
        # request.Data.block = encrypted_block
        # for member in nodes:
        #     channel = grpc.insecure_channel(member, options=options)
        #     stub = grpc_pb2_grpc.ConsensusStub(channel)
        #     try:
        #         response = stub.BroadBlock(request)
        #         print(response.Result)
        #     except Exception as e:
        #         print("CONNECTION FAILED IN BroadBlock PHASE!")
        #         print("Exception details:", e)
        #         # PREPARE_flag=False
        #         break

        print("开始等待其余委员会的备选区块传播")
        while(len(block_list) != self.committee_size):
            time.sleep(2)
            print(len(block_list))
            # print(block_list)
            continue
        print("=========================开始传送私钥=========================")
        t = threading.Thread(target=self.Key_send(key, iv))
        t.start()
        t.join()
        # 私钥作为发送的消息
        # request_2 = grpc_pb2.Privatemessage()
        # request_2.Data.node_id = self.node_id
        # # pri_key = pickle.dumps(private_key)
        # # private_key_pem = private_key.private_bytes(
        # #     encoding=serialization.Encoding.PEM,
        # #     format=serialization.PrivateFormat.PKCS8,
        # #     encryption_algorithm=serialization.NoEncryption()  # 不加密
        # # )
        # request_2.Data.key = key
        # request_2.Data.iv = iv
        # self_node = set()
        # self_node.add(p2p.SELF_IP_PORT)
        # nodes = self.committee - self_node
        # for member in nodes:
        #     channel = grpc.insecure_channel(member, options=options)
        #     stub = grpc_pb2_grpc.ConsensusStub(channel)
        #     try:
        #         response = stub.BroadKey(request)
        #         print(response.Result)
        #     except Exception as e:
        #         print("CONNECTION FAILED IN BroadKey PHASE!")
        #         print("Exception details:", e)
        #         # PREPARE_flag=False
        #         break
        # key_list.append({"node_id": self.node_id, "key": key, "iv": iv})
        print("开始等待其余委员会的私钥传播")
        while(len(key_list) != self.committee_size):
            time.sleep(2)
            continue
        print("=========================开始传送最终区块=========================")
        # 解密并统计区块
        # decrypted_blocks = []
        block_counter = {}

        for block_entry in block_list:
            node_id = block_entry["node_id"]
            encrypted_block = block_entry["block"]
            # encrypted_block = pickle.loads(encrypted_block_bytes)
            private_key_entry = next((key_entry for key_entry in key_list if key_entry["node_id"] == node_id), None)
            if private_key_entry is None:
                print(f"No matching private key found for node_id {node_id}")
                continue

            id_key = private_key_entry["key"]
            id_iv = private_key_entry["iv"]

            try:
                # block_bytes = decrypt_block(encrypted_block, private_key)
                block_bytes = decrypt_data_block(id_iv, encrypted_block, id_key)
                block = pickle.loads(block_bytes)
                # decrypted_blocks.append(block)
                block_hash = hash_block(block)

                if block_hash in block_counter:
                    block_counter[block_hash]["count"] += 1
                else:
                    block_counter[block_hash] = {"block": block, "count": 1}
            except Exception as e:
                print("Decryption failed for a block:", e)

        # 选出频次最高的区块
        max_count = 0
        block_list.clear()
        key_list.clear()
        for block_hash, info in block_counter.items():
            if info["count"] > max_count:
                max_count = info["count"]
                self.final_block = info["block"]

        if self.final_block:
            print("最终选定的区块成功")
            time.sleep(5)
            self.broadcast_final_block()
        else:
            print("未能选定最终区块。")


    def Block_send(self, encrypted_block):
        global block_list
        self_node = set()
        self_node.add(self.ipport)
        nodes = self.committee - self_node
        print("BroadBlock 阶段, 目标节点有", nodes)
        # 加密区块作为发送的消息
        request = grpc_pb2.Blockmessage()
        # print("node_id的类型是",type(self.node_id))
        request.node_id = self.node_id
        request.block = encrypted_block
        for member in nodes:
            channel = grpc.insecure_channel(member, options=options)
            stub = grpc_pb2_grpc.ConsensusStub(channel)
            # print(member)
            try:
                response = stub.BroadBlock(request)
                print(response.Result)
            except Exception as e:
                print("CONNECTION FAILED IN BroadBlock PHASE!")
                print("Exception details:", e)
                # PREPARE_flag=False
                break
        block_list.append({"node_id": self.node_id, "block": request.block})

    def Key_send(self, key, iv):
        request_2 = grpc_pb2.Privatemessage()
        request_2.node_id = self.node_id
        request_2.key = key
        request_2.iv = iv
        self_node = set()
        self_node.add(self.ipport)
        nodes = self.committee - self_node
        for member in nodes:
            channel = grpc.insecure_channel(member, options=options)
            stub = grpc_pb2_grpc.ConsensusStub(channel)
            try:
                response = stub.BroadKey(request_2)
                print(response.Result)
            except Exception as e:
                print("CONNECTION FAILED IN BroadKey PHASE!")
                print("Exception details:", e)
                # PREPARE_flag=False
                break
        key_list.append({"node_id": self.node_id, "key": key, "iv": iv})

    def broadcast_final_block(self):
        # 广播最终区块由委员会中端口号最小的成员执行
        if self.ipport == min(self.committee, key=lambda node: int(node.split(":")[1])):
            self_node = set()
            self_node.add(self.ipport)
            # print(self_node)
            nodes = self.nodes - self_node
            request = grpc_pb2.FinalBlockMessage()
            request.node_id = self.node_id
            request.block.CopyFrom(self.final_block)
            for node in nodes:
                print(nodes)
                channel = grpc.insecure_channel(node, options=options)
                stub = grpc_pb2_grpc.ConsensusStub(channel)
                try:
                    response = stub.BroadFinalBlock(request)
                    print(f"Final block broadcast to {node}: {response.Result}")
                except Exception as e:
                    print(f"Failed to broadcast final block to {node}: {e}")
        else:
            print("非最低端口号的成员无需广播最终区块。")

    def receive_new_block(self):
        global final_block_receive
        print("进入 receive_new_block")
        if self.ipport == min(self.committee, key=lambda node: int(node.split(":")[1])):
            print("发送区块的Leader不需要接收新区快")
            final_block_receive = self.final_block
            self.add_block(self.final_block)
        else:
            print(min(self.committee, key=lambda node: int(node.split(":")[1])))
            while(final_block_receive == None):
                time.sleep(2)
                continue
            self.add_block(final_block_receive)

# Consensus using gRPC
class Consensus(grpc_pb2_grpc.ConsensusServicer):
    def BroadBlock(self, request, context):
        global block_list
        print("it is BroadBlock")
        # print(request.Data)
        block_list.append({"node_id": request.node_id, "block": request.block})
        return grpc_pb2.ConsensusRsp(Result='BLock Received Successfully')
        
    def BroadKey(self, request, context):
        global key_list
        print("it is BroadKey")
        # loaded_private_key = serialization.load_pem_private_key(
        #     request.data.key,
        #     password=None
        # )

        key_list.append({"node_id": request.node_id, "key": request.key, "iv": request.iv})
        return grpc_pb2.ConsensusRsp(Result='key Received Successfully')


    def BroadFinalBlock(self, request, context):
        global final_block_receive
        print("it is BroadFinalBlock")
        final_block_receive = request.block
        return grpc_pb2.ConsensusRsp(Result='New Block Received Successfully')
    
    # def ExchangeGrad(self, request, context):
    #     global grad_list
    #     print("收到来自节点的梯度。")
    #     grad_list.append(request)
    #     return grpc_pb2.TensorReceive(Result='Grad Received Successfully')
    