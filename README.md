Repository for university projects for 2026 spring semester.
# ML

## Directory for university tasks about machine learning

# ML_task

## Directory for machine learining task assigned for lack of ML model in the project

# Lindström prject

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and go to `http://localhost:5000`

## Map Format

Maps are stored as JSON files in the `maps/` directory:

```json
{
    "name": "warehouse_a",
    "dimensions": [17, 18],
    "shelves": {
        "A": [
            {"code": 1, "coords": [2, 3]},
            {"code": 2, "coords": [2, 4]}
        ],
        "B": [
            {"code": 1, "coords": [5, 3]},
            {"code": 2, "coords": [5, 4]}
        ]
    }
}
```

## Order Format

```json
{
    "id": "ORD-001",
    "map": "warehouse_a",
    "items": ["A001XXX", "B002YYY"],
    "worker": "WRK-001",
    "status": "pending",
    "priority": "normal",
    "deadline": "2026-04-15T12:00:00",
    "created_at": "2026-04-10T12:00:00",
    "updated_at": "2026-04-10T12:00:00"
}
```

## Item Code Format

Item codes follow the format `GCCCIIII` where:
- `G` — Shelf group letter (e.g. `A`, `B`)
- `CCC` — Shelf section code (e.g. `001`, `002`)
- `IIII` — Product/item identifier (e.g. `XXX`, `ABC123`)

Example: `A001XXX` = Shelf group A, section 1, product XXX

## API Endpoints

### Maps
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/list-maps` | List all saved maps |
| GET | `/load-map/<name>` | Load a specific map |
| POST | `/save-map` | Save a map |
| DELETE | `/delete-map/<name>` | Delete a map |

### Orders
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/orders` | List all orders |
| POST | `/orders` | Create a new order |
| GET | `/orders/<id>` | Get a specific order |
| PUT | `/orders/<id>` | Update an order |
| DELETE | `/orders/<id>` | Delete an order |
| GET | `/api/orders/<id>/preview` | Preview order route on map |

### Workers
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/workers` | List all workers |
| POST | `/api/workers` | Add a new worker |
| GET | `/api/workers/<id>` | Get a specific worker |
| PUT | `/api/workers/<id>` | Update a worker |
| DELETE | `/api/workers/<id>` | Delete a worker |

### Algorithm
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/solve` | Run pathfinding algorithm |

## Database

The system uses SQLite with SQLAlchemy ORM. To migrate to PostgreSQL in the future, 
update the connection string in `config.py`:

```python
SQLALCHEMY_DATABASE_URI = "postgresql://user:pass@localhost/warehouse"
```
