import streamlit as st
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# Graph, Use Case: Emergency Supply Robot
locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6)
}

hospital_graph = {
    "Pharmacy": {
        "Main_Corridor": 2.2,
        "Patient_Wing": 4.1
    },
    "Main_Corridor": {
        "Nursing_Station": 2.2
    },
    "Patient_Wing": {
        "Laboratory": 5.0
    },
    "Nursing_Station": {
        "Laboratory": 3.2,
        "Emergency_Ward": 6.0
    },
    "Laboratory": {
        "Emergency_Ward": 3.2
    },
    "Emergency_Ward": {}
}

# Heuristic
def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Path reconstruction
def reconstruct_path(came_from, current):
    path = []

    while current is not None:
        path.append(current)
        current = came_from[current]

    path.reverse()
    return path

# GBFS
def gbfs(start, goal):
    came_from = {
        start: None
    }
    frontier = []
    expanded = []
    heapq.heappush(
        frontier,
        (heuristic(start, goal), start)
    )

    while frontier:
        current = heapq.heappop(frontier)[1]

        if current == goal:
            break

        if current in expanded:
            continue

        expanded.append(current)

        for neighbor in hospital_graph[current]:
            if neighbor not in expanded:
                if neighbor not in came_from:
                    came_from[neighbor] = current

                heapq.heappush(
                    frontier,
                    (heuristic(neighbor, goal), neighbor)
                )

    if goal not in came_from:
        return None, 0

    path = reconstruct_path(came_from, goal)
    cost = 0

    for i in range(len(path) - 1):
        cost += hospital_graph[path[i]][path[i + 1]]

    return path, cost

# A*
def a_star(start, goal):
    came_from = {
        start: None
    }
    g_cost = {
        start: 0
    }
    frontier = []
    expanded = []

    heapq.heappush(
        frontier,
        (heuristic(start, goal), start)
    )

    while frontier:
        current = heapq.heappop(frontier)[1]

        if current == goal:
            break

        if current in expanded:
            continue

        expanded.append(current)

        for neighbor in hospital_graph[current]:
            new_cost = g_cost[current] + hospital_graph[current][neighbor]

            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                came_from[neighbor] = current

                f_cost = new_cost + heuristic(neighbor, goal)

                heapq.heappush(
                    frontier,
                    (f_cost, neighbor)
                )

    if goal not in came_from:
        return None, 0

    path = reconstruct_path(came_from, goal)
    cost = g_cost[goal]

    return path, cost

##########################################
# Streamlit GUI Code
# Set Page Config
st.set_page_config(
    page_title="Emergency Supply Robot",
    page_icon="🏥",
    layout="wide"
)

# write meaningful title and description for the app
st.title("Emergency Supply Robot")
st.write("Find a path between hospital locations using GBFS or A* Search.")

# define the nodes and their coordinates
nodes = list(hospital_graph.keys())

# create a selectbox for the user to choose the start and goal nodes
start = st.selectbox(
    "Select Initial Node",
    nodes,
    index=nodes.index("Pharmacy")
)

goal = st.selectbox(
    "Select Goal Node",
    nodes,
    index=nodes.index("Emergency_Ward")
)

# create a selectbox for the user to choose the search algorithm
algorithm = st.selectbox(
    "Select Search Algorithm",
    ["GBFS", "A*"]
)

if st.button("Run Search"):
    if algorithm == "GBFS":
        # run the GBFS algorithm with the selected start and goal nodes
        path, cost = gbfs(start, goal)
    else:
        # run the A* algorithm with the selected start and goal nodes
        path, cost = a_star(start, goal)

    if path is None:
        # display a error message indicating that no path was found
        st.error("No path was found between the selected nodes.")
    else:
        # Display result
        st.subheader("Search Result")
        st.write(
            f"Algorithm: {algorithm}"
        )
        st.write(
            f"Solution Path: {' → '.join(path)}"
        )
        st.write(
            f"Total Path Cost: {cost:.2f}"
        )

        # Visualize NetworkX graph
        G = nx.DiGraph()

        for node, neighbors in hospital_graph.items():
            for neighbor, weight in neighbors.items():
                G.add_edge(
                    node,
                    neighbor,
                    weight=weight
                )

        pos = locations

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        # WRITE REMAINING NETWORKX VISUALIZATION CODE HERE
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_size=2000,
            arrows=True
        )

        path_edges = list(zip(path, path[1:]))

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=path_edges,
            edge_color="red",
            width=3
        )

        ax.set_title(
            f"{algorithm} Solution Path"
        )
        ax.axis("off")
        st.pyplot(fig)