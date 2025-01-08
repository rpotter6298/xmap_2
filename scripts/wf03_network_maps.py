from classes import graphnet, xmap_qc
import pickle
import json
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
graphnet_keys = json.load(open(Path("keys", ".keychain")))['graphnet']

grapher = graphnet.Graphnet(uri=graphnet_keys['uri'], username=graphnet_keys['username'], password=graphnet_keys['password'])

# Load in proteins of interest
with open ("data/top_correlation_proteins.pickle", "rb") as f:
    correlation_dict = pickle.load(f)
with open("data/study.pickle", "rb") as f:
    study = pickle.load(f)
qc = xmap_qc(study.normalized_data)
study_proteins = qc.filter_low_variance_proteins(threshold=1e-5).columns
study_proteins = [protein.split("_")[0] for protein in study_proteins]
#load in cluster proteins
with open("data/cluster_1_proteins.pickle", "rb") as f:
    cluster_1_proteins = pickle.load(f)
with open("data/cluster_2_proteins.pickle", "rb") as f:
    cluster_2_proteins = pickle.load(f)


def construct_protein_network(grapher, proteins, min_strength=0.5, threshold=None):
    """
    Executes a query to retrieve shortest paths and constructs a NetworkX graph.

    Parameters:
    - grapher: A Graphnet instance connected to Neo4j.
    - proteins: List of protein IDs of interest.
    - min_strength: Minimum strength for relationships to consider in the path.
    - threshold: Degree threshold for identifying important nodes. If None, defaults to (0.5 * len(proteins)) - 1.

    Returns:
    - G: A NetworkX graph of the protein interaction network.
    - important_nodes: Dictionary of nodes with degrees above the threshold.
    """
    # Set default threshold if not specified
    if threshold is None:
        threshold = max(0, int((0.5 * len(proteins)) - 1))  # Ensure non-negative threshold

    query = """
        UNWIND $proteins AS p1
        UNWIND $proteins AS p2
        WITH p1, p2 WHERE p1 <> p2
        MATCH path = shortestPath((a:protein {protein_ID: p1})-[r:combined_score*..12]-(b:protein {protein_ID: p2}))
        WHERE ALL(rel IN relationships(path) WHERE rel.strength >= $min_strength)
        RETURN a.protein_ID AS protein1, b.protein_ID AS protein2, nodes(path) AS path_nodes, relationships(path) AS path_rels
    """
    parameters = {
        "proteins": proteins,
        "min_strength": min_strength
    }
    results = grapher.execute_query(query, parameters=parameters)

    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes and edges to the graph based on query results
    for record in results:
        path_nodes = record["path_nodes"]
        for i in range(len(path_nodes) - 1):
            G.add_edge(path_nodes[i]["protein_ID"], path_nodes[i + 1]["protein_ID"])

    # Calculate degrees and identify important nodes
    node_degrees = dict(G.degree())
    important_nodes = {node: degree for node, degree in node_degrees.items() if degree > threshold}

    # Output the important nodes and their degrees
    print(f"Threshold for high-degree nodes: {threshold}")
    print("Nodes appearing in multiple shortest paths (above threshold):")
    for node, degree in important_nodes.items():
        print(f"{node}: {degree}")

    return G, important_nodes

def visualize_protein_network(G, MPI, start_nodes, study_proteins, threshold=3, figsize=(12, 10), file_path=None):
    """
    Visualizes the protein interaction network with shortest paths highlighted.

    Parameters:
    - G: A NetworkX graph of the protein interaction network.
    - MPI: Dictionary of nodes with degrees above the threshold (Multiple Path Intersections).
    - start_nodes: List of starting protein IDs of interest (highlighted in orange).
    - study_proteins: List of protein IDs to highlight with a green ring.
    - threshold: Degree threshold for highlighting nodes that appear in multiple shortest paths.
    - figsize: Size of the figure for visualization.
    - file_path: Path to save the visualization as a file. If None, displays the plot.

    Returns:
    - None
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    highlighted_nodes = set(start_nodes)
    study_proteins_set = set(study_proteins)

    # Set node colors, sizes, and edge colors for visualization
    node_colors = []
    node_sizes = []
    node_edgecolors = []
    for node in G.nodes():
        if node in highlighted_nodes:
            color = "orange"
            size = 600
        elif node in MPI:
            color = "red"
            size = 800
        else:
            color = "skyblue"
            size = 400
        node_colors.append(color)
        node_sizes.append(size)

        # Set edge color to green if the node is in study_proteins; otherwise, no edge
        if node in study_proteins_set:
            edgecolor = "greenyellow"
        else:
            edgecolor = 'none'  # No edge color
        node_edgecolors.append(edgecolor)

    # Plot the network
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="gray",
        font_size=8,
        font_weight="bold",
        edgecolors=node_edgecolors,
        linewidths=2,
    )

    # Create custom legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Start Nodes (Highlighted Proteins)', 
               markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='MPI (High-Degree Proteins)', 
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Other Proteins', 
               markerfacecolor='skyblue', markersize=10),
        Line2D([0], [0], marker='o', markeredgecolor='greenyellow', markerfacecolor='white', label='Proteins included in study',
               markersize=10, linewidth=2),
    ]

    plt.legend(handles=legend_elements, loc='best')
    plt.title("Protein Interaction Network with Study Proteins Highlighted")

    # Save or show the plot
    if file_path:
        plt.savefig(file_path, format='png', bbox_inches='tight')
        print(f"Visualization saved to {file_path}")
    else:
        plt.show()

def visualize_protein_network(G, MPI, start_nodes, study_proteins, threshold=3, figsize=(12, 10), file_path=None):
    """
    Visualizes the protein interaction network with appropriate colors, shapes, sizes, and a single combined legend.

    Parameters:
    - G: A NetworkX graph of the protein interaction network.
    - MPI: Dictionary of nodes with degrees above the threshold (Multiple Path Intersections).
    - start_nodes: List of starting protein IDs of interest (always orange triangles).
    - study_proteins: List of protein IDs to highlight with a different shape (triangles).
    - threshold: Degree threshold for highlighting nodes that appear in multiple shortest paths.
    - figsize: Size of the figure for visualization.
    - file_path: Path to save the visualization as a file. If None, displays the plot.

    Returns:
    - None
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    # Ensure only valid nodes are considered
    valid_study_proteins = [node for node in study_proteins if node in G.nodes()]
    highlighted_nodes = set(start_nodes)
    study_proteins_set = set(valid_study_proteins)
    MPI_nodes = set(MPI.keys()) - highlighted_nodes  # Exclude start nodes from MPI nodes

    # Separate nodes into categories
    other_nodes = set(G.nodes()) - highlighted_nodes - MPI_nodes

    # Define positions for nodes
    pos = nx.spring_layout(G)

    plt.figure(figsize=figsize)

    # Plot Start Nodes (Always Orange Triangles)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(highlighted_nodes),
        node_color="orange",
        node_size=2000,  # Increased size
        node_shape="^",  # Triangle shape
        edgecolors="black",  # Thin black outline
        linewidths=0.5,
        label="Start Nodes (Highlighted Proteins)",
    )

    # Plot Study Proteins That Are MPI (Red Triangles)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(study_proteins_set & MPI_nodes),
        node_color="red",
        node_size=2400,  # Increased size
        node_shape="^",  # Triangle shape
        edgecolors="black",  # Thin black outline
        linewidths=0.5,
        label="MPI in Study",
    )

    # Plot Study Proteins That Are Neither Start Nodes Nor MPI (Skyblue Triangles)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(study_proteins_set - highlighted_nodes - MPI_nodes),
        node_color="skyblue",
        node_size=1600,  # Increased size
        node_shape="^",  # Triangle shape
        edgecolors="black",  # Thin black outline
        linewidths=0.5,
        label="Other Study Proteins",
    )

    # Plot Other MPI Nodes (Red Circles)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(MPI_nodes - study_proteins_set),
        node_color="red",
        node_size=2400,  # Increased size
        node_shape="o",  # Circle shape
        edgecolors="black",  # Thin black outline
        linewidths=0.5,
        label="MPI (High-Degree Proteins)",
    )

    # Plot Other Proteins (Skyblue Circles)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(other_nodes - study_proteins_set),
        node_color="skyblue",
        node_size=1600,  # Increased size
        node_shape="o",  # Circle shape
        edgecolors="black",  # Thin black outline
        linewidths=0.5,
        label="Other Proteins",
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # Create a combined legend
    legend_elements = [
        # Colors
        Rectangle((0, 0), 1, 1, color="skyblue", label="Other Proteins"),
        Rectangle((0, 0), 1, 1, color="orange", label="Start Nodes"),
        Rectangle((0, 0), 1, 1, color="red", label="MPI Nodes"),
        # Shapes
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black", markeredgecolor="black", markersize=12, label="Included in Study"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markeredgecolor="black", markersize=12, label="Not Included in Study"),

    ]

    # Add the combined legend
    plt.legend(
        handles=legend_elements,
        loc="upper right",
        title="Legend",
        ncol=2,  # Two columns for shapes and colors
        bbox_to_anchor=(1, 1),  # Position legend in the upper-right corner
        fontsize=14,
        title_fontsize=16,
    )

    plt.title("Protein Interaction Network with Study Proteins Highlighted")

    # Save or show the plot
    if file_path:
        plt.savefig(file_path, format="png", bbox_inches="tight")
        print(f"Visualization saved to {file_path}")
    else:
        plt.show()


G, MPI = construct_protein_network(grapher, proteins=correlation_dict["Proteins"])
visualize_protein_network(G, MPI=MPI,  start_nodes=correlation_dict["Proteins"], study_proteins=study_proteins)
print("STMN4" in G.nodes())


G_cluster_2, MPI_cluster_2 = construct_protein_network(grapher, proteins=cluster_2_proteins)
visualize_protein_network(G_cluster_2, MPI=MPI_cluster_2,  start_nodes=cluster_2_proteins, study_proteins=study_proteins)

G_cluster_1, MPI_cluster_1 = construct_protein_network(grapher, proteins=cluster_1_proteins)
visualize_protein_network(G_cluster_1, MPI=MPI_cluster_1,  start_nodes=cluster_1_proteins, study_proteins=study_proteins)

G_se, MPI_se = construct_protein_network(grapher, proteins=correlation_dict["SE"])
visualize_protein_network(G_se, MPI=MPI_se,  start_nodes=correlation_dict["SE"], study_proteins=study_proteins)

G_vfi, MPI_vfi = construct_protein_network(grapher, proteins=correlation_dict["VFI_Loss"])
visualize_protein_network(G_vfi, MPI=MPI_vfi,  start_nodes=correlation_dict["VFI_Loss"], study_proteins=study_proteins)

G_iop, MPI_iop = construct_protein_network(grapher, proteins=correlation_dict["IOP_Diagnosis"])
visualize_protein_network(G_iop, MPI=MPI_iop,  start_nodes=correlation_dict["IOP_Diagnosis"], study_proteins=study_proteins)

G_MD, MPI_MD = construct_protein_network(grapher, proteins=correlation_dict["MD_Diagnosis"])
visualize_protein_network(G_MD, MPI=MPI_MD,  start_nodes=correlation_dict["MD_Diagnosis"], study_proteins=study_proteins)

