from flask import Blueprint, request, jsonify
from extensions import db
from models import Order
from datetime import datetime
import json

orders_bp = Blueprint("orders", __name__)

def genererate_order_id():
    last = Order.query.order_by(Order.id.desc()).first()
    if not last:
        return "ORD-001"
    num = int(last.id.split("-")[1]) + 1
    return f"ORD-{num:03d}"

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