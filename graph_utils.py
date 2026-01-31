
import networkx as nx
import pandas as pd
from datetime import timedelta

def get_k_hop_subgraph(G, center_node, k=1, max_nodes=2000):
    """
    Extracts a k-hop subgraph centered around a specific node.
    Limits the number of nodes to avoid UI crashes.
    """
    if center_node not in G:
        return None
        
    # BFS to find neighbors up to k hops
    # We use a custom BFS because we want to include both successors (downstream) and predecessors (upstream)
    # usually for money laundering tracing. Or maybe just successors? 
    # The requirement says "Surfing" - implies following the flow. 
    # But usually investigators want to see where money came FROM too.
    # NetworkX ego_graph collects neighbors within radius.
    
    # undirected=True effectively treats edges as bidirectional for neighborhood search,
    # finding both sources and destinations.
    print(f"Extracting {k}-hop subgraph for {center_node}...")
    subgraph = nx.ego_graph(G, center_node, radius=k,  undirected=True)
    
    if len(subgraph.nodes()) > max_nodes:
        print(f"Subgraph too large ({len(subgraph.nodes())} nodes), clipping...")
        # Simple clipping: keep center + first (max_nodes-1) neighbors
        nodes_to_keep = list(subgraph.nodes())[:max_nodes]
        if center_node not in nodes_to_keep:
            nodes_to_keep[0] = center_node
        subgraph = subgraph.subgraph(nodes_to_keep)
        
    return subgraph

def filter_edges_by_time(G, time_window_hours=None, end_time=None):
    """
    Filters edges in the graph that fall within a time window.
    Note: NetworkX edges can have attributes.
    """
    if time_window_hours is None:
        return G
        
    # This is tricky on an existing Graph object because modifying it in place is destructive.
    # Better to filter the DataFrame before creating the graph if possible, 
    # OR create a new graph view.
    # Given the requirement "Time-Based Surfing", dynamic filtering is needed.
    
    # If G is small (subgraph), iteration is fine.
    sub_edges = []
    
    # Assuming end_time is the reference point (e.g., current time or latest tx)
    # If end_time is not provided, take the max timestamp in the graph?
    # For now, let's assume the UI passes a valid end_time or we don't filter relative to "now".
    
    # Actually, simpler: The UI will filter the DataFrame of transactions, then we rebuild the subgraph.
    # Rebuilding a huge graph is bad. Rebuilding a small subgraph is fine.
    pass

def trace_path(G, start_node, end_node):
    """
    Finds simple paths between two nodes.
    MAX_PATH_LENGTH limited to avoid infinite loops or long computation.
    """
    try:
        paths = list(nx.all_simple_paths(G, source=start_node, target=end_node, cutoff=5))
        return paths
    except Exception as e:
        print(f"Error tracing path: {e}")
        return []
