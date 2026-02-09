from cmath import inf
import heapq
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

W, H = 17, 18

# 0 = free, 1 = obstacle
grid = [[0]*W for _ in range(H)]

    
for x in range(2, 15, 3):
    for y in range(2, 14):
        grid[y][x] = 1

# Add shelves
for a in range(2,15):
    grid[8][a] = 0
    grid[16][a] = 1
    
start = (0,0) # Starting point 
items = [(13, 9), (3,10), (10, 3), (6, 6), (6, 15), (15,7), (15,16)] # Items

# visualize items in grid
for x, y in items:
    grid[y][x] = 7

matrix = np.array(grid)
print("Matrix shape:", matrix.shape)
print(matrix)

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def neighbors(node):
    x, y = node
    for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
        if 0 <= nx < W and 0 <= ny < H:
            if grid[ny][nx] != 1:
                yield (nx, ny)


def astar(start, goal):
    pq = [(0, start)]
    g = {start: 0}
    parent = {}

    while pq:
        _, current = heapq.heappop(pq)

        if current == goal:
            return g[goal]

        for nb in neighbors(current):
            new_cost = g[current] + 1
            if nb not in g or new_cost < g[nb]:
                g[nb] = new_cost
                f = new_cost + heuristic(nb, goal)
                heapq.heappush(pq, (f, nb))

    return -1


# Distance matrix
points = [start] + items
dist = {(a,b): astar(a,b) for a in points for b in points if a != b}


# TSP
best_order = None
min_total = float(inf)

for perm in itertools.permutations(items):
    total = 0
    curr = start
    for p in perm:
        total += dist[(curr, p)]
        curr = p
    if total < min_total:
        min_total = total
        best_order = perm
    
# for m in dist:
#      print(m)

print("\nOptimal Order:", best_order)
print("Total Distance:", min_total)

# Define colors for vaules
colors = ['white', 'lightblue', 'orange'] # 0=empty, 1=sotrage, 7=pickUp
bins = 3 # one bin for possible value
cmap = mcolors.ListedColormap(colors[:bins])
bounds = np.arange(bins + 1)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(10, 12))
im = plt.imshow(matrix, cmap=cmap, norm=norm, interpolation='none')
plt.colorbar(im, ticks=bounds[:-1], boundaries=bounds, label='Occupancy (0=empty, 1=storage, 7=pickUp)')
plt.title('Warehouse Layout Heatmap')
plt.xlabel('Columns (A-Q)')
plt.ylabel('Rows (1-18)')
plt.grid(True, alpha=0.3)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        if val != 0:
            plt.text(j, i, str(matrix[i,j]), ha='center', va='center', fontsize=8)
plt.show()
