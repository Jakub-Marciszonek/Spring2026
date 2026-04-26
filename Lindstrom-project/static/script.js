// map editor variables
let W = 17;
let H = 18;
const size = 40;

let mode = "item";
let start = [0, 0];
let items = [];
let shelves = [];
let route = [];

let isPainting = false;
let painted = new Set();

let tooltipTimeout = null;

// map editor popup variables
let selectedShelves = [];
let isSelecting = false;

// orders edit popup variables
let editingOrderId = null;

//  workers edit popupo variables
let editingWorkerId = null;

// settings variables
let animationSpeed = 60;

// map editor elements
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");

const shelfBtn = document.getElementById("shelfBtn");
const itemBtn = document.getElementById("itemBtn");
const runBtn = document.getElementById("runBtn");

const widthInput = document.getElementById("WidthInput");
const heightInput = document.getElementById("HeightInput");

const clearBtn = document.getElementById("clearBtn");

const mapName = document.getElementById("mapName");
const mapSelect = document.getElementById("mapSelect");

const deleteMapBtn = document.getElementById("deleteMapBtn");
const saveBtn = document.getElementById("saveBtn");

const shelfTooltip = document.getElementById("shelfTooltip")

// map editor popup elements
const shelfPopup = document.getElementById("shelfPopup");
const shelfGroup = document.getElementById("shelfGroup");
const shelfDirection = document.getElementById("shelfDirection");
const shelfStart = document.getElementById("shelfStart");
const shelfStep = document.getElementById("shelfStep");
const shelfPopupConfirm = document.getElementById("shelfPopupConfirm");
const shelfPopupCancel = document.getElementById("shelfPopupCancel");

// order elements
const createOrderBtn = document.getElementById("createOrderBtn");
const orderMap = document.getElementById("orderMap");
const orderItems = document.getElementById("orderItems");
const orderPriority = document.getElementById("orderPriority");
const orderDeadline = document.getElementById("orderDeadline");
const ordersTableBody = document.getElementById("ordersTableBody");

// order craete popup elements
const orderPopup = document.getElementById("orderPopup");
const orderPopupConfirm = document.getElementById("orderPopupConfirm");
const orderPopupCancel = document.getElementById("orderPopupCancel");

// order edit popup elements
const editOrderPopup = document.getElementById("editOrderPopup");
const editOrderStatus = document.getElementById("editOrderStatus");
const editOrderWorker = document.getElementById("editOrderWorker");
const editOrderPriority = document.getElementById("editOrderPriority");
const editOrderDeadline = document.getElementById("editOrderDeadline");
const editOrderConfirm = document.getElementById("editOrderConfirm");
const editOrderCancel = document.getElementById("editOrderCancel");

// wokrers elements
const createWorkerBtn = document.getElementById("createWorkerBtn");
const workersTableBody = document.getElementById("workersTableBody");

// workers create popup
const workerPopup = document.getElementById("workerPopup");
const workerName = document.getElementById("workerName");
const workerStatus = document.getElementById("workerStatus");
const workerPopupConfirm = document.getElementById("workerPopupConfirm");
const workerPopupCancel = document.getElementById("workerPopupCancel");

// Settings elements
const themeBtn = document.getElementById("themeBtn");
const speedSlider = document.getElementById("speedSlider");
const speedLabel = document.getElementById("speedLabel");

speedLabel.textContent = `${animationSpeed}ms`

// navitagion
document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.onclick = () => {
        document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(`view-${btn.dataset.view}`).classList.add("active");
    };
});

// map editor - map drawing
canvas.addEventListener("mousedown", (e) => {
    if (e.button === 2) { // right click
        isSelecting = true;
        selectedShelves = [];
        const rect = canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / size);
        const y = Math.floor((e.clientY - rect.top) / size);
        const shelf = shelves.find(s => s.coords[0] === x && s.coords[1] === y);
        if (shelf) {
            selectedShelves.push(shelf);
        }
        drawGrid();
        return;
    }
    // left click
    isPainting = true;
    painted = new Set();
    handleClick(e);
});

canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / size);
    const y = Math.floor((e.clientY - rect.top) / size);
    const shelf = shelves.find(s => s.coords[0] === x && s.coords[1] === y);
    
    clearTimeout(tooltipTimeout);
    shelfTooltip.classList.add("hidden");
    if (shelf && shelf.group) {
        tooltipTimeout = setTimeout(() => {
            shelfTooltip.textContent = `Shelf ${shelf.group} - Code: ${shelf.code ?? "unassigned"}`;
            shelfTooltip.style.left = `${e.clientX + 12}px`;
            shelfTooltip.style.top = `${e.clientY + 12}px`;
            shelfTooltip.classList.remove("hidden");
        }, 600);
    }

    if (isSelecting && e.buttons === 2) {
        if (shelf && !selectedShelves.includes(shelf)) {
            selectedShelves.push(shelf);
            drawGrid();
        }
        return;
    }
    if (!isPainting) {
        return;
    }   

    const key = `${x}, ${y}`;
    if (painted.has(key)) {
        return;
    }
    painted.add(key);
    handleClick(e);
});

canvas.addEventListener("mouseup", (e) => {
    if (e.button === 2) {
        isSelecting = false;
        if (selectedShelves.length > 0) {
            shelfGroup.value = selectedShelves[0].group || "";
            shelfDirection.value = selectedShelves[0].direction || "lr";
            shelfStart.value = selectedShelves[0].start || 1;
            shelfStep.value = selectedShelves[0].step || 1;
            shelfPopup.classList.remove("hidden");
        }
        return;
    }
    isSelecting = false;
    isPainting = false;
    painted = new Set();
});

canvas.addEventListener("mouseleave", () => {
    isSelecting = false;
    isPainting = false;
    painted = new Set();
});

widthInput.onchange = () => {
    W = Math.min(30, Math.max(5, parseInt(widthInput.value) || 17));
    widthInput.value = W;
    canvas.width = W * size;
    items = items.filter(p => p[0] < W && p[1] < H);
    shelves = shelves.filter(s => s.coords[0] < W && s.coords[1] < H);
    drawGrid();
}

heightInput.onchange = () => {
    H = Math.min(30, Math.max(5, parseInt(heightInput.value) || 18));
    heightInput.value = H;
    canvas.height = H * size;
    items = items.filter(p => p[0] < W && p[1] < H);
    shelves = shelves.filter(s => s.coords[0] < W && s.coords[1] < H);
    drawGrid();
}

clearBtn.onclick = () => {
    if (!confirm("Clear all cells?")) {
        return;
    }
    items = [];
    shelves = [];
    drawGrid();
}

//map editor - save/load/delete
async function fetchMaps() {
    const res = await fetch("/list-maps");
    const data = await res.json();
    mapSelect.innerHTML = '<option value="">Load map...</option>';
    data.maps.forEach(name => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        mapSelect.appendChild(option);
    });
}

mapSelect.onchange = async () => {
    if (!mapSelect.value) {
        shelves = [];
        items = [];
        drawGrid();
        return;
    }
    const res = await fetch(`/load-map/${mapSelect.value}`);
    const data = await res.json();

    W = data.dimensions[0];
    H = data.dimensions[1];

    if (Array.isArray(data.shelves)) {
        shelves = data.shelves.map(s =>
            Array.isArray(s) ? { coords: s, group: null, code: null } : s // handle flat format
            // of map json file (example below) and current format
            /*
            "shelves": [
                [5, 1], 
                [6, 1], 
            ]
            */
        );
    } else {
        const loadedShelves = [];
        Object.entries(data.shelves).forEach(([group, cell]) => {
            cell.forEach(cell => {
                loadedShelves.push({ 
                    coords: cell.coords,
                    group: group === "unassigned" ? null : group,
                    code: cell.code,
                    direction: cell.direction || "lr",
                    start: cell.start || 1,
                    step: cell.step || 1
                });
            });
        });
        shelves = loadedShelves;
    }

    widthInput.value = W;
    heightInput.value = H;
    canvas.width = W * size;
    canvas.height = H * size;
    mapName.value = mapSelect.value;
    drawGrid();
};

deleteMapBtn.onclick = async () => {
    if (!mapSelect.value) {
        alert("Please select a map to delete");
        return;
    }
    if (!confirm(`Delete "${mapSelect.value}"?`)) {
        return;
    }
    const res = await fetch(`/delete-map/${mapSelect.value}`, { method: "DELETE" });
    const data = await res.json();
    if (data.status === "ok") {
        mapSelect.querySelector(`option[value="${mapSelect.value}"]`).remove();
        mapSelect.value = "";
        shelves = [];
        drawGrid();
    }
}

saveBtn.onclick = async () => {
    const name = mapName.value.trim();
    if (!name) {
        alert("Please enter a map name");
        return;
    }
    if (!confirm(`Save map as "${name}"?
        \nThis will overwrite any existing map with the same name.`)) {
        return;
    }
    const res = await fetch("/save-map", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            name,
            dimensions: [W, H],
            shelves: shelvesToJSON()
            }),
    });
    const data = await res.json();
    if (data.status === "ok" ) {
        saveBtn.textContent = "Saved!";
        setTimeout(() => saveBtn.textContent = "Save Map", 1500);
        await fetchMaps();
        mapSelect.value = name;
    }
};

function shelvesToJSON() {
    const groups = {};
    shelves.forEach(s => {
        const key = s.group || "unassigned";
        if (!groups[key]) groups[key] = [];
        groups[key].push({
            code: s.code,
            coords: s.coords 
        });
    });
    return groups;
}

function setMode(newMode) {
    mode = newMode;
    shelfBtn.classList.toggle("active", mode === "shelf");
    itemBtn.classList.toggle("active", mode === "item");
}
shelfBtn.onclick = () => setMode("shelf");
itemBtn.onclick = () => setMode("item");

function handleClick(e) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / size);
    const y = Math.floor((e.clientY - rect.top) / size);

    if (mode === "item") {
        const isShelf = shelves.some((s) => s.coords[0] === x && s.coords[1] === y);
        if (isShelf) { // can't place item on shelf
            flashCell(x, y);
            return;
        } 

        const idx = items.findIndex((i) => i[0] === x && i[1] === y);
        if (idx !== -1) {
            items.splice(idx, 1); // remove item if already exists
        } else {
            items.push([x, y]);
        }
    } else if (mode === "shelf") {
        const isItem = items.some((s) => s[0] === x && s[1] === y);
        if (isItem) { // can't place item on shelf
            flashCell(x, y);
            return;
        } 

        const idx = shelves.findIndex((s) => s.coords[0] === x && s.coords[1] === y);
        if (idx !== -1) {
            shelves.splice(idx, 1); // remove shelf if already exists
        } else {
            shelves.push({ coords: [x, y], group: null, code: null});
        }
    }

    drawGrid();
}

function recalculateCodes(group) {
    const groupShelves = shelves.filter(s => s.group === group);
    const direction = groupShelves[0]?.direction || "lr";
    const start = groupShelves[0]?.start || 1;
    const step = groupShelves[0]?.step || 1;

    groupShelves.sort((a, b) => {
        if (direction === "lr") {
            return a.coords[0] - b.coords[0] || a.coords[1] - b.coords[1]; // left right > top bottom
        } else if (direction === "rl") {
            return b.coords[0] - a.coords[0] || a.coords[1] - b.coords[1]; // right left > top bottom
        } else if (direction === "tb") {
            return a.coords[1] - b.coords[1] || a.coords[0] - b.coords[0]; // top bottom > left right
        } else if (direction === "bt") {
            return b.coords[1] - a.coords[1] || a.coords[0] - b.coords[0]; // bottom top > left roght
        }
    });

    groupShelves.forEach((s, i) => {
        s.code = start + i * step;
    });
}

//map editor - right click popup
shelfPopupConfirm.onclick = () => {
    if (!selectedShelves.length) {
        return;
    }
    const group = shelfGroup.value.trim().toUpperCase();
    if (!group) {
        alert("Please enter a group name");
        return;
    }
    selectedShelves.forEach(s => {
        s.group = group;
        s.direction = shelfDirection.value;
        s.start = parseInt(shelfStart.value) || 1;
        s.step = parseInt(shelfStep.value) || 1;
    });
    recalculateCodes(group);
    shelfPopup.classList.add("hidden");
    selectedShelves = [];
    drawGrid();
};

shelfPopupCancel.onclick = () => {
    shelfPopup.classList.add("hidden");
    selectedShelves = [];
    drawGrid();
}

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

// map editor - grid rendering
function flashCell(x, y) {
    ctx.fillStyle = "rgba(239, 68, 68, 0.5)";
    ctx.fillRect(x * size, y * size, size, size);
    setTimeout(() => drawGrid(), 300);
}

function drawGrid() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            ctx.fillStyle = "#111827";
            if (selectedShelves.some(s => s.coords[0] === x && s.coords[1] === y)) {
                ctx.fillStyle = "#3b82f6";
            } else if (shelves.some((s) => s.coords[0] === x && s.coords[1] === y)) {
                ctx.fillStyle = "#374151";
            }
            ctx.fillRect(x * size, y * size, size, size);
            ctx.strokeStyle = "#1f2937";
            ctx.strokeRect(x * size, y * size, size, size);
        }
    }

    drawItems();
}

function drawItems() {
    ctx.fillStyle = "orange";
    items.forEach((p) => {
        ctx.beginPath();
        ctx.arc(
            p[0] * size + size / 2,
            p[1] * size + size / 2,
            8,
            0,
            Math.PI * 2
        );
        ctx.fill();
    });

    ctx.fillStyle = "lime";
    ctx.beginPath();
    ctx.arc(
        start[0] * size + size / 2,
        start[1] * size + size / 2,
        10,
        0,
        Math.PI * 2
    );
    ctx.fill();
}

// map editor - algorithm
async function solve() {
    const res = await fetch("/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ // sends flat coords
            start,
            items,
            shelves: shelves.map(s => s.coords), // unpacks shelves coords
            W,
            H
        }), 
    });

    const data = await res.json();
    route = data.path;
    animate();
}

// map editor - animation
async function animate() {
    for (let i = 0; i < route.length; i++) {
        drawGrid();

        ctx.strokeStyle = "cyan";
        ctx.lineWidth = 3;
        ctx.beginPath();

        for (let j = 0; j <= i; j++) {
            const r = route[j];
            if (j === 0) {
                ctx.moveTo(
                    r[0] * size + size / 2,
                    r[1] * size + size / 2
                );
            } else {
                ctx.lineTo(
                    r[0] * size + size / 2,
                    r[1] * size + size / 2
                );
            }
        }
        ctx.stroke();

        const p = route[i];
        ctx.fillStyle = "cyan";
        ctx.beginPath();
        ctx.arc(
            p[0] * size + size / 2,
            p[1] * size + size / 2,
            10,
            0,
            Math.PI * 2
        );
        ctx.fill();
        await new Promise((r) => setTimeout(r, animationSpeed));
    }
}

// orders
async function fetchOrders() {
    const res = await fetch("/orders");
    const orders = await res.json();
    console.log("orders:", orders);
    renderOrders(orders);
}

function renderOrders(orders) {
    if (orders.length === 0) {
        ordersTableBody.innerHTML = '<tr><td colspan="8">No orders yet.</td></tr>';
        return;
    }
    ordersTableBody.innerHTML = orders.map(o => `
        <tr>
            <td>${o.id}</td>
            <td>${o.map}</td>
            <td>${o.items.join(", ")}</td>
            <td><span class="status-badge status-${o.status}">${o.status}</span></td>
            <td class="priority-${o.priority}">${o.priority}</td>
            <td>${o.deadline ? new Date(o.deadline).toLocaleString() : "-"}</td>
            <td>${o.worker || "-"}</td>
            <td>
                <button onclick="previewOrder('${o.id}')">Preview</button>
                <button onclick="editOrder('${o.id}')">Edit</button>
                <button onclick="cancelOrder('${o.id}')" ${o.status === "cancelled" ? "disabled" : ""}>Cancel</button>
            </td>
        </tr>
    `).join("");
}

async function cancelOrder(id) {
    if (!confirm(`Cancel order ${id}?`)) return;
    await fetch(`/orders/${id}`, {
        method: "PUT",
        headers: {"Content-Type": "application/json" },
        body: JSON.stringify({ status: "cancelled" })
    });
    fetchOrders();
}

async function populateOrderMaps() {
    const res = await fetch("/list-maps");
    const data = await res.json();
    orderMap.innerHTML = `<option value="">Select map...</option>` + 
    data.maps.map(m => `<option value="${m}">${m}</option>`).join("");
}

createOrderBtn.onclick = async () => {
    await populateOrderMaps();
    orderPopup.classList.remove("hidden");
};

async function previewOrder(id) {
    const res = await fetch(`/api/orders/${id}/preview`);
    const data = await res.json();

    if (data.errors.length > 0) {
        alert("Some items could not be resolved:\n" + data.errors.join("\n"));
    }

    W = data.map.dimensions[0];
    H = data.map.dimensions[1];
    shelves = data.map.shelves ? Object.entries(data.map.shelves).flatMap(([group, cells]) =>
        cells.map(c => ({ coords: c.coords, group, code: c.code }))
    ) : [];
    items = data.items;

    document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    document.querySelector("[data-view='editor']").classList.add("active");
    document.getElementById("view-editor").classList.add("active");

    canvas.width = W * size;
    canvas.height = H * size;
    drawGrid();

    solve();
}

// order create popup
orderPopupCancel.onclick = () => {
    orderPopup.classList.add("hidden");
    orderItems.value = ""
}

async function validateItems(items, mapName) {
    const res = await fetch(`/load-map/${mapName}`);
    const mapData = await res.json();
    const errors = [];

    items.forEach(item => {
        const match = item.match(/^([A-Za-z]+)(\d+)/);
        if (!match) {
            errors.push(`"${item}" - invalid format, must start with letters then numbers`);
            return;
        }

        const group = match[1].toUpperCase();
        const code = parseInt(match[2]);

        if (!mapData.shelves[group]) {
            errors.push(`"${item}" - shelf group "${group}" not found in map`);
            return;
        }

        const shelf = mapData.shelves[group].find(s => s.code === code);
        if (!shelf) {
            errors.push(`"${item}" - code not found in shelf group "${group}"`);
        }
    });

    return errors;
}

orderPopupConfirm.onclick = async () => {
    const items = orderItems.value.trim().split("\n").map(s => s.trim()).filter(s => s);
    if (!items.length) { 
        alert("Please enter at least one item"); 
    return;
    }
    if (!orderMap.value) {
        alert("Please select a map"); 
        return;
    } 
    const errors = await validateItems(items, orderMap.value);
    if (errors.length > 0) {
        alert("Invalid items:\n\n" + errors.join("\n"));
        return;
    }

    const res = await fetch("/orders", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            map: orderMap.value,
            items,
            priority: orderPriority.value,
            deadline: orderDeadline.value || null
        })
    });
    if (res.ok) {
        orderPopup.classList.add("hidden");
        orderItems.value = "";
        fetchOrders();
    }
};

//order edit popup
async function editOrder(id) {
    fetch(`/orders/${id}`)
        .then(res => res.json())
        .then(async order => {
            editingOrderId = id;
            await populateWorkerSelect();
            editOrderStatus.value = order.status;
            editOrderWorker.value = order.worker || "";
            editOrderPriority.value = order.priority;
            editOrderDeadline.value = order.deadline ? order.deadline.slice(0, 16) : "";
            editOrderPopup.classList.remove("hidden");
        });
}

async function populateWorkerSelect() {
    const res = await fetch("/workers");
    const workers = await res.json();
    editOrderWorker.innerHTML = `<option value="">Unassigned</option>` +
    workers.map(w => `<option value="${w.id}">${w.name}</option>`).join("");
}

editOrderConfirm.onclick = async () => {
    if (!editingOrderId) {
        return;
    }
    await fetch(`/orders/${editingOrderId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            status: editOrderStatus.value,
            worker: editOrderWorker.value || null,
            priority: editOrderPriority.value,
            deadline: editOrderDeadline.value || null
        })
    });
    editOrderPopup.classList.add("hidden");
    editingOrderId = null;
    fetchOrders();
};

editOrderCancel.onclick = () => {
    editOrderPopup.classList.add("hidden");
    editingOrderId = null;
};

// workers

async function fetchWorkers() {
    const res = await fetch("/workers");
    const workers = await res.json();
    renderWorkers(workers);
}

function renderWorkers(workers) {
    if (workers.length === 0) {
        workersTableBody.innerHTML = '<tr><td colspan="5">No workers yet.</td></tr>';
        return;
    }
    workersTableBody.innerHTML = workers.map(w => `
        <tr>
            <td>${w.id}</td>
            <td>${w.name}</td>
            <td><span class="status-badge status-${w.status}">${w.status}</span></td>
            <td>${new Date(w.created_at).toLocaleString()}</td>
            <td>
                <button onclick="editWorker('${w.id}')">Edit</button>
                <button onclick="deleteWorker('${w.id}')">Delete</button>
            </td>
        </tr>
    `).join("");
}

async function addWorker() {
    const name = workerName.value.trim();
    if (!name) {
        alert("Please enter a worker name");
        return;
    }

    const res = await fetch("/workers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, status: workerStatus.value })
    });

    if (res.ok) {
        workerPopup.classList.add("hidden");
        fetchWorkers();
    }
}

async function deleteWorker(id) {
    if (!confirm(`Delete worker ${id}?`)) return;
    await fetch(`/workers/${id}`, { method: "DELETE" });
    fetchWorkers();
}

// workers create popup
createWorkerBtn.onclick = () => {
    editingWorkerId = null;
    workerName.value ="";
    workerStatus.value = "available";
    workerPopupConfirm.textContent = "Add";
    workerPopup.classList.remove("hidden")
};

workerPopupCancel.onclick = () => {
    workerPopup.classList.add("hidden");
    editingWorkerId = null;
    workerPopupCancel.textContent = "Cancel";
};


workerPopupConfirm.onclick = async () => {
    const name = workerName.value.trim();
    if (!name) {
        alert("Please enter a worker name");
        return;
    }

    if (editingWorkerId) {
        await fetch(`/workers/${editingWorkerId}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, status: workerStatus.value })
        });
    } else {
        await fetch("/workers", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, status: workerStatus.value })
        });
    }

    editingWorkerId = null;
    workerPopup.classList.add("hidden");
    fetchWorkers();
};

//workers edit popup
async function editWorker(id) {
    const res = await fetch(`/workers/${id}`);
    const worker = await res.json();

    editingWorkerId = id;
    workerName.value = worker.name;
    workerStatus.value = worker.status;
    workerPopupConfirm.textContent = "Save";
    fetchWorkers()
    workerPopup.classList.remove("hidden");
}

// settings
themeBtn.onclick = () => {
    document.body.classList.toggle("light");
    const isLight = document.body.classList.contains("light");
    themeBtn.textContent = isLight ? "Dark Mode" : "Light Mode";
    localStorage.setItem("theme", isLight ? "light" : "dark");
};

speedSlider.oninput = () => {
    animationSpeed = parseInt(speedSlider.value);
    speedLabel.textContent = `${animationSpeed}ms`;
};

if (localStorage.getItem("theme") === "light") {
    document.body.classList.add("light");
    themeBtn.textContent = "Dark Mode";
}

// map editor init
fetchMaps();

widthInput.value = W;
heightInput.value = H;

animationSpeed = parseInt(speedSlider.value);
speedLabel.textContent = `${animationSpeed}ms`;

canvas.width = W * size;
canvas.height = H * size;

itemBtn.classList.add("active");
runBtn.onclick = solve;
drawGrid();

// orders init
fetchOrders();

// workers init

fetchWorkers();