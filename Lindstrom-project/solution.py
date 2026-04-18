import heapq

W, H = 17, 18

grid = [[0] * W for _ in range(H)]
for x in range(2, 15, 3):
    for y in range(2, 14):
        grid[y][x] = 1

for a in range(2, 15):
    grid[8][a] = 0
    grid[16][a] = 1


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(node):
    x, y = node
    for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
        if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] != 1:
            yield (nx, ny)


def astar(start, goal, grid):
    W = len(grid[0])
    H = len(grid)

    pq = [(0, start)]
    g = {start: 0}
    parent = {}

    while pq:
        _, current = heapq.heappop(pq)

        if current == goal:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path

        x, y = current
        for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            if 0 <= nx < W and 0 <= ny < H and grid[ny][nx] != 1:
                nb = (nx, ny)
                new_cost = g[current] + 1
                if nb not in g or new_cost < g[nb]:
                    g[nb] = new_cost
                    parent[nb] = current
                    f = new_cost + heuristic(nb, goal)
                    heapq.heappush(pq, (f, nb))
    return None


def compute_path(start, items, shelves, W, H):
    grid = [[0] * W for _ in range(H)]

    # build shelves
    for x, y in shelves:
        if 0 <= x < W and 0 <= y < H:
            grid[y][x] = 1

    remaining = items.copy()
    curr = start
    order = []

    while remaining:
        best = None
        best_len = float("inf")

        for p in remaining:
            path = astar(curr, p, grid)
            if path is None:
                continue
            if len(path) < best_len:
                best = p
                best_len = len(path)

        if best is None:
            break

        order.append(best)
        remaining.remove(best)
        curr = best

    full = []
    curr = start
    for p in order:
        seg = astar(curr, p, grid)
        if seg is None:
            continue
        if full:
            seg = seg[1:]
        full += seg
        curr = p

    return full
