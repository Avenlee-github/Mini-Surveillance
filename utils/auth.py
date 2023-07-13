from functools import wraps
from flask import Flask, request, Response
import yaml

# 读取config.yaml文件
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

authentication = config['authentication']
username = authentication['username']
password = authentication['password']

def check_auth(username, password):
    """Check if the provided username and password are correct."""
    correct_username = username
    correct_password = password
    return username == correct_username and password == correct_password

def authenticate():
    """Send a 401 response that enables basic auth."""
    return Response("Could not verify your access level for that URL.\nYou have to login with proper credentials.", 401, {"WWW-Authenticate": "Basic realm='Login Required'"})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)

    return decorated