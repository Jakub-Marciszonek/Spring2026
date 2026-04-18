import os

class Config:
    SQLALCHEMY_DATABASE_URI = "sqlite:///warehouse.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAPS_DIR = os.path.join(os.path.dirname(__file__), "maps")