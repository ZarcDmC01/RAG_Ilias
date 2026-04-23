from flask import Flask, jsonify, request, redirect, url_for

from flasgger import Swagger

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")

app.config['SWAGGER'] = {
    'title': "RAG² API",
    'uiversion': 3,
    'description': "Interface de prédiction des prix avec sécurité JWT",
    'version': "1.0.0",
    "securityDefinitions": {
        "Bearer": {
            "type": "apiKey",
            "name": "Authorization",
            "in": "header",
            "description": "Entre: Bearer <ton_token>"
        }
    }
}
swagger = Swagger(app)

