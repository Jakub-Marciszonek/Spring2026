from flask import Blueprint, request, jsonify
from extensions import db
from models import Order
from datetime import datetime
from config import Config
import json, re, os

orders_bp = Blueprint("orders", __name__)

def genererate_order_id():
    last = Order.query.order_by(Order.id.desc()).first()
    if not last:
        return "ORD-001"
    num = int(last.id.split("-")[1]) + 1
    return f"ORD-{num:03d}"

def find_adjacent_walkable(coord, map_data):
    x, y = coord
    W, H = map_data["dimensions"]

    all_shelf_coords = [
        c["coords"] for cells in map_data["shelves"].values() for c in cells
    ]

    for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
        if 0 <= nx < W and 0 <= ny < H and [nx, ny] not in all_shelf_coords:
            return [nx, ny]
    
    return None

@orders_bp.route("/orders", methods=["GET"])
def get_orders():
    orders = Order.query.all()
    return jsonify([o.to_dict() for o in orders])

@orders_bp.route("/orders", methods=["POST"])
def create_order():
    data = request.json
    order = Order(
        id=genererate_order_id(),
        map=data["map"],
        items=json.dumps(data["items"]),
        worker=data.get("worker"),
        status="pending",
        priority=data.get("priority", "normal"),
        deadline=data.get("deadline")
    )
    db.session.add(order)
    db.session.commit()
    return jsonify(order.to_dict()), 201

@orders_bp.route("/orders/<id>", methods=["GET"])
def get_order(id):
    order = Order.query.get_or_404(id)
    return jsonify(order.to_dict())

@orders_bp.route("/orders/<id>", methods=["PUT"])
def update_order(id):
    order = Order.query.get_or_404(id)
    data = request.json
    if "status" in data:
        order.status = data["status"]
    if "worker" in data:
        order.worker = data["worker"]
    if "priority" in data:
        order.priority = data["priority"]
    if "deadline" in data:
        order.deadline = data["deadline"]
    order.updated_at = datetime.utcnow().isoformat()
    db.session.commit()
    return jsonify(order.to_dict())

@orders_bp.route("/orders/<id>", methods=["DELETE"])
def delete_order(id):
    order = Order.query.get_or_404(id)
    db.session.delete(order)
    db.session.commit()
    return jsonify({ "status": "ok" })

@orders_bp.route("/api/orders/<id>/preview", methods=["GET"])
def preview_order(id):
    try:
        order = Order.query.get_or_404(id)
        
        map_path = os.path.join(Config.MAPS_DIR, f"{order.map}.json")
        with open(map_path) as f:
            map_data = json.load(f)
        
        items = json.loads(order.items)
        coords = []
        errors = []
        
        for item in items:
            match = re.match(r'^([A-Za-z]+)(\d+)', item)
            if not match:
                errors.append(f"{item} — invalid format")
                continue
            group = match.group(1).upper()
            code = int(match.group(2))
            shelves = map_data["shelves"].get(group, [])
            shelf = next((s for s in shelves if s["code"] == code), None)
            if not shelf:
                errors.append(f"{item} — not found in map")
                continue
            adjacent = find_adjacent_walkable(shelf["coords"], map_data)
            if adjacent:
                coords.append(adjacent)
            else:
                errors.append(f"{item} — no walkable cell adjacent to shelf")
        
        return jsonify({
            "map": map_data,
            "items": coords,
            "errors": errors
        })
    except Exception as e:
        print(f"Preview error: {e}")
        return jsonify({"error": str(e)}), 500
