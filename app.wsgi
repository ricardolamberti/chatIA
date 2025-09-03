import sys
import os

sys.path.append('C:\\dev\\python\\chatdatabase')

from flask_app import app as application
application.secret_key = 'anythingwished'