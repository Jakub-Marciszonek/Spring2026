from flask import Blueprint, request, jsonify, render_template
from config import Config
from solution import compute_path
import os, json

editor_bp = Blueprint("editor", __name__)
map_dir = Config.MAPS_DIR

@editor_bp.route("/")
def home():
    return render_template("index.html")

@editor_bp.route("/solve", methods=["POST"])
def solve():
    data = request.json
    start = tuple(data["start"])
    items = [tuple(i) for i in data["items"]]
    shelves = [tuple(i) for i in data["shelves"]]
    W = data.get("W", 17)
    H = data.get("H", 18)

    path = compute_path(start, items, shelves, W, H)
    return jsonify({"path": path})

@editor_bp.route("/list-maps", methods=["GET"])
def list_maps():
    os.makedirs(map_dir, exist_ok=True)
    maps = [f.replace(".json", "") for f in os.listdir(map_dir) if f.endswith(".json")]
    return jsonify({"maps": maps})

@editor_bp.route("/load-map/<name>", methods=["GET"])
def laod_map(name):
    path = os.path.join(map_dir, f"{name}.json")
    with open(path) as f:
        return jsonify(json.load(f))
    
@editor_bp.route("/save-map", methods=["POST"])
def save_map():
    data = request.json
    name = data.get("name", "map").strip().replace(" ", "_")
    os.makedirs(os.path.dirname(map_dir), exist_ok=True)
    with open(os.path.join(map_dir, f"{name}.json"), "w") as f:
        json.dump(data, f)
    return jsonify({"status": "ok"})
    
@editor_bp.route("/delete-map/<name>", methods=["DELETE"])
def delete_map(name):
    path = os.path.join(map_dir, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)
        return jsonify({ "status": "ok"})
    return jsonify({ "status": "error", "message": "Map not found" }), 404