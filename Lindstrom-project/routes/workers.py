from flask import Blueprint, jsonify, request
from extensions import db
from models import Worker
from datetime import datetime

workers_bp = Blueprint("workers", __name__)

def generate_worker_id():
    last = Worker.query.order_by(Worker.id.desc()).first()
    if not last:
        return "WRK-001"
    num = int(last.id.split("-")[1]) + 1
    return f"WRK-{num:03d}"

@workers_bp.route("/workers", methods=["GET"])
def get_workers():
    workers = Worker.query.all()
    return jsonify([{
        "id": w.id, 
        "name": w.name, 
        "status": w.status,
        "created_at": w.created_at
        } for w in workers])

@workers_bp.route("/workers", methods=["POST"])
def create_worker():
    data = request.json
    worker = Worker(
        id=generate_worker_id(),
        name=data["name"],
        status=data.get("status", "available")
    )
    db.session.add(worker)
    db.session.commit()
    return jsonify({"id": worker.id, "name": worker.name, "status": worker.status}), 201

@workers_bp.route("/workers/<id>", methods=["GET"])
def get_worker(id):
    worker = Worker.query.get_or_404(id)
    return jsonify({"id": worker.id, "name": worker.name, "status": worker.status, "created_at": worker.created_at})

@workers_bp.route("/workers/<id>", methods=["PUT"])
def update_worker(id):
    worker = Worker.query.get_or_404(id)
    data = request.json
    if "name" in data:
        worker.name = data["name"]
    if "status" in data:
        worker.status = data["status"]
    db.session.commit()
    return jsonify({"id": worker.id, "name": worker.name, "status": worker.status})

@workers_bp.route("/workers/<id>", methods=["DELETE"])
def delete_worker(id):
    worker = Worker.query.get_or_404(id)
    db.session.delete(worker)
    db.session.commit()
    return jsonify({"status": "ok"})