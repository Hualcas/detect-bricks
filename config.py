# config.py
#import os
#
#class Config:
#    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key'
#    MONGODB_SETTINGS = {
#        'db': 'databas',
#        'host': 'localhost',
#        'port': 27017
#    }

import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    MONGODB_SETTINGS = {
        'host': os.environ.get('MONGODB_URI')  # <- Render lo toma de variables de entorno
    }
