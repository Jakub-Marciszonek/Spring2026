from flask import Flask
from models import Order
from config import Config
from extensions import db
from routes.editor import editor_bp
from routes.orders import orders_bp
from routes.workers import workers_bp

def create_app():

    app = Flask(__name__)

    app.config.from_object(Config)
    db.init_app(app)

    app.register_blueprint(editor_bp)
    app.register_blueprint(orders_bp)
    app.register_blueprint(workers_bp)

    with app.app_context():
        db.create_all()
        
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
