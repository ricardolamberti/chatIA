from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def create_app():
    app = Flask(__name__)
 
    db.init_app(app)

    from .public import public_bp
    app.register_blueprint(public_bp)

    return app