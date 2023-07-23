import os
import json
import base64
import aes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from llm import LLM

LLM()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/v1/model':
            self.send_response(200)
            self.end_headers()
            llm = LLM()
            response = json.dumps({
                'result': llm.model_name()
            })

            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_length).decode('utf-8'))

        if self.path == '/api/v1/generate':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            prompt = body['prompt']
            llm = LLM()
            response = json.dumps({
                'results': [{
                    'text': llm.create_completion(prompt, 50)
                }]
            })
            self.wfile.write(response.encode('utf-8'))
        elif self.path == '/api/v2/generate':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            key = os.getenv("PSK")
            key = key.encode()
            prompt = aes.str_decrypt(body['prompt'], key)

            llm = LLM()
            response = json.dumps({
                'results': [{
                    'text':  aes.str_encrypt(llm.create_completion(prompt, 50), key)
                }]
            })
            self.wfile.write(response.encode('utf-8'))
        elif self.path == '/api/v1/token-count':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            prompt = body['prompt']
            llm = LLM()
            response = json.dumps({
                'results': [{
                    'tokens': llm.token_count(prompt)
                }]
            })
            self.wfile.write(response.encode('utf-8'))
        elif self.path == '/api/v2/token-count':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            key = os.getenv("PSK")
            key = key.encode()
            prompt = aes.str_decrypt(body['prompt'], key)
            llm = LLM()
            response = json.dumps({
                'results': [{
                    'tokens': llm.token_count(prompt)
                }]
            })
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)


if __name__ == '__main__':
    host = os.getenv("SERVER_HOST")
    if host is None:
        host="0.0.0.0"
    port = os.getenv("SERVER_PORT")
    if port is None:
        port=5000
    else:
        port = int(port)
    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()


