from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from routes.dataset import dataset_routes
from routes.evaluation import evaluation_routes
from routes.dashboard import dashboard_routes
from routes.developer_report import developer_report_routes
from routes.regulator_report import regulator_report_routes
from routes.comparison import comparison_routes
from routes.admin import admin_routes
from routes.executive_report import executive_report_routes
from routes.remediation import remediation_routes
from services.model_loader import load_artifacts

import os


app = Flask(__name__)

try:
    load_artifacts()
    print("✅ Pre-trained model loaded successfully")
except FileNotFoundError as e:
    print(f"⚠️  {e} — will fall back to on-the-fly training")

# ── CORS ──────────────────────────────────────────────────────────────────────
CORS(app,
     origins="*",
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=False,
     automatic_options=True)

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        res = app.make_default_options_response()
        res.headers["Access-Control-Allow-Origin"]  = "*"
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        res.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        res.headers["Access-Control-Max-Age"]       = "3600"
        return res

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response

# ── Blueprints ────────────────────────────────────────────────────────────────
app.register_blueprint(dataset_routes)
app.register_blueprint(evaluation_routes)
app.register_blueprint(dashboard_routes)
app.register_blueprint(developer_report_routes)
app.register_blueprint(regulator_report_routes)
app.register_blueprint(comparison_routes)
app.register_blueprint(admin_routes)

# ── New Round 3 blueprints ────────────────────────────────────────────────────
app.register_blueprint(executive_report_routes)
app.register_blueprint(remediation_routes)

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "upload.html")

@app.route("/<page>.html")
def serve_page(page):
    return send_from_directory(FRONTEND_DIR, f"{page}.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "3.0", "round": "JEDI Round 3"}), 200

if __name__ == "__main__":
    print(f"\n  Frontend : {FRONTEND_DIR}")
    print(f"  Open     : http://127.0.0.1:5000/upload.html\n")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False
    )