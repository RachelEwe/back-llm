import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

def str_encrypt(message, key):
    if isinstance(message, str):
        message = message.encode()
    if isinstance(key, str):
        key = key.encode()
    result = aes_encrypt(message, key)

    return base64.b64encode(result).decode()

def str_decrypt(message, key):
    if isinstance(message, str):
        message = message.encode()
    if isinstance(key, str):
        key = key.encode()
    result = aes_decrypt(base64.b64decode(message), key)

    return result.decode()

def aes_encrypt(message, key):
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()
    padded_message = padder.update(message) + padder.finalize()
    ciphertext = encryptor.update(padded_message) + encryptor.finalize()

    return iv + ciphertext

def aes_decrypt(ciphertext, key):
    iv = ciphertext[:16]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_message = decryptor.update(ciphertext[16:]) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    unpadded_message = unpadder.update(decrypted_message) + unpadder.finalize()

    return unpadded_message
