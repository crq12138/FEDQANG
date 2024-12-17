from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from ecdsa import SigningKey, VerifyingKey, NIST384p
import pickle
from cryptography.hazmat.primitives import serialization

# 生成 RSA 密钥对（用于加密/解密）
def generate_rsa_keys():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    public_key = private_key.public_key()
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()  # 不加密
    )
    print(type(private_key_pem))
    loaded_private_key = serialization.load_pem_private_key(
        private_key_pem,
        password=None  # 无加密密码
    )
    private_key = loaded_private_key
    return private_key, public_key

# 生成 ECDSA 密钥对（用于签名/验证）
def generate_ecdsa_keys():
    priv_key = SigningKey.generate(curve=NIST384p)
    pub_key = priv_key.get_verifying_key()
    return priv_key, pub_key

# 使用公钥加密 block
def encrypt_block(block, public_key):
    block_bytes = pickle.dumps(block)
    encrypted_block = public_key.encrypt(
        block_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return encrypted_block

# 使用私钥解密 block
def decrypt_block(encrypted_block, private_key):
    block_bytes = private_key.decrypt(
        encrypted_block,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    block = pickle.loads(block_bytes)
    return block

# 使用私钥签名加密的 block
def sign_encrypted_block(encrypted_block, priv_key):
    signature = priv_key.sign(encrypted_block)
    return signature

# 验证签名
def verify_signature(encrypted_block, signature, pub_key):
    try:
        pub_key.verify(signature, encrypted_block)
        print("Signature is valid!")
        return True
    except Exception as e:
        print(f"Invalid signature: {e}")
        return False

# # 测试完整流程
# if __name__ == "__main__":
#     # 生成密钥对
#     rsa_private_key, rsa_public_key = generate_rsa_keys()
#     ecdsa_private_key, ecdsa_public_key = generate_ecdsa_keys()

#     # 创建一个区块
#     block = {"data": "Sensitive Block Data", "index": 1}

#     # 加密区块
#     encrypted_block = encrypt_block(block, rsa_public_key)

#     # 签名加密的区块
#     signature = sign_encrypted_block(encrypted_block, ecdsa_private_key)

#     # 传输 signed_block
#     signed_block = {
#         'encrypted_block': encrypted_block,
#         'signature': signature
#     }

#     # 解密区块并验证签名
#     received_encrypted_block = signed_block['encrypted_block']
#     received_signature = signed_block['signature']

#     # 验证签名
#     if verify_signature(received_encrypted_block, received_signature, ecdsa_public_key):
#         # 签名有效，解密区块
#         decrypted_block = decrypt_block(received_encrypted_block, rsa_private_key)
#         print("Decrypted block:", decrypted_block)
#     else:
#         print("Failed to verify signature.")


item = set('asd', 'afxzv')