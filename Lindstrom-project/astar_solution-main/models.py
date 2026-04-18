from datetime import datetime
from extensions import db
import json

class Order(db.Model):
    __tablename__ = "orders"
    id = db.Column(db.String, primary_key=True)
    map = db.Column(db.String, nullable=False)
    items = db.Column(db.String, nullable=False)
    worker = db.Column(db.String, nullable=True)
    status = db.Column(db.String, default="pending")
    priority = db.Column(db.String, default="normal")
    deadline = db.Column(db.String, nullable=True)
    created_at = db.Column(db.String, default=lambda: datetime.utcnow().isoformat())
    updated_at = db.Column(db.String, default=lambda: datetime.utcnow().isoformat())

    def to_dict(self):
        return {
            "id": self.id,
            "map": self.map,
            "items": json.loads(self.items),
            "worker": self.worker,
            "status": self.status,
            "priority": self.priority,
            "deadline": self.deadline,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
class Worker(db.Model):
    __tablename__ = "workers"
    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String, nullable=False)
    status = db.Column(db.String, default="Available") # available, busy, offline
    created_at = db.Column(db.String, default=lambda: datetime.utcnow().isoformat())