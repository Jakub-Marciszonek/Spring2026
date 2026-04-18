## Problem solving for the course "Project with Company" at LAB 

### Pipeline

Warehouse Grid -> User place selves & items -> Algorithm -> Path planning -> Simulation -> Result

### Implementation

1. Warehouse Grid: Presented by a 2D grid
  - 0 = free space
  - 1 = shelves

2. Adjust the "start" in script.js to change the initial position of the user.

3. Algorithm:
  - Pathfinding: A* algorithm with complexity  O(n!)
  - Heuristic: Manhattan distance (4 directions instead of diagonal movement)
  - Greedy Nearest Neighbor: Route planner with complexity O(n^2)

### Explaination

1. Why choosing A* algorithm?
  - Optimality
  - Efficiency: Faster than Dijkstra's
  - Manhattan heuristic for grid movement

2. Why choosing Greedy Nearest Neighbor?
  - Simplicity: Easy to implement
  - Efficiency: Faster than exact TSP solutions
  - Suitable for small item sets

3. Limitations:
  - Crazy high complexity for A* and GNN combination in a larger grid that may lead to performance issues and crash.


### How to use

1. Clone the repository
```python
git clone 
```
2. Install Flask and run app.py to start the server
```python
pip install flask
python app.py
```
3. Open http://localhost:5000

4. Mukava päivä!


### Contributor
1. Mira Vorne for giving me the wonderful project.
2. ChatGPT for FE generated and JS convert from python.