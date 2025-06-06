from flask import Flask # type: ignore
from .extensions import login_manager, bcrypt
from config import Config
from mongoengine import connect # type: ignore
from app.utils import descargar_modelo, descargar_imagenes  # 👈

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # ✅ Conexión MongoEngine usando los valores de config
    connect(**app.config['MONGODB_SETTINGS'])

    # Inicializar extensiones
    login_manager.init_app(app)
    bcrypt.init_app(app)

    from .models import User
    from .routes import routes as routes_blueprint
    from .auth import auth as auth_blueprint
    app.register_blueprint(routes_blueprint)
    app.register_blueprint(auth_blueprint, url_prefix='/auth')

    try:
        descargar_modelo()
        descargar_imagenes()
    except Exception as e:
        print(f"[ERROR] No se pudo descargar el modelo: {e}")
    

    @login_manager.user_loader
    def load_user(user_id):
        return User.objects(id=user_id).first()

    return app
