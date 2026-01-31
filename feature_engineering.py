
import pandas as pd
import networkx as nx
import numpy as np
from datetime import datetime
import config

def load_data():
    """Loads transactions and labels."""
    print("Loading data...")
    transactions_df = pd.read_csv(config.TRANSACTIONS_PATH)
    labels_df = pd.read_csv(config.LABELS_PATH)
    
    # Ensure timestamp is datetime
    transactions_df['Timestamp'] = pd.to_datetime(transactions_df['Timestamp'])
    return transactions_df, labels_df

def build_graph(transactions_df):
    """Builds a MultiDiGraph from transactions."""
    G = nx.from_pandas_edgelist(
        transactions_df, 
        source='Source_Wallet_ID', 
        target='Dest_Wallet_ID', 
        edge_attr=['Amount', 'Timestamp'],
        create_using=nx.MultiDiGraph()
    )
    return G

def compute_features(transactions_df, labels_df):
    """Computes wallet-level features for the model."""
    print("Building graph...")
    G = build_graph(transactions_df)
    
    print("Calculating structural features...")
    # Initialize features dictionary
    features = {}
    
    # Pre-calculate degrees to avoid repeated lookups
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    # Iterate through all nodes in the graph
    for node in G.nodes():
        node_feats = {}
        
        # 1. Fan-in / Fan-out
        f_in = in_degree.get(node, 0)
        f_out = out_degree.get(node, 0)
        node_feats['fan_in'] = f_in
        node_feats['fan_out'] = f_out
        
        # 2. Total Amounts
        # Get incoming edges
        in_edges = G.in_edges(node, data=True)
        in_amounts = [d['Amount'] for _, _, d in in_edges]
        total_in = sum(in_amounts)
        
        # Get outgoing edges
        out_edges = G.out_edges(node, data=True)
        out_amounts = [d['Amount'] for _, _, d in out_edges]
        total_out = sum(out_amounts)
        
        node_feats['total_in'] = total_in
        node_feats['total_out'] = total_out
        
        # 3. Unique Counterparties
        unique_sources = set([u for u, _, _ in in_edges])
        unique_dests = set([v for _, v, _ in out_edges])
        node_feats['unique_sources'] = len(unique_sources)
        node_feats['unique_dests'] = len(unique_dests)
        
        # 4. Mule Score (Fan-in * Fan-out)
        # Normalized logic can be applied later, raw for now
        node_feats['mule_score'] = f_in * f_out
        
        # 5. Amount Similarity (Balance Ratio)
        # Avoid division by zero
        if total_in > 0:
            node_feats['balance_ratio'] = total_out / total_in
        else:
            node_feats['balance_ratio'] = 0.0
            
        # 6. Time Behavior (Burstiness)
        # Average time gap for incoming and outgoing
        # This is expensive for massive graphs, so we'll do a simplified version if needed.
        # For now, let's skip complex time variance per node to keep it fast for the hackathon prototype,
        # or implement a simple "burst" heuristic: standard deviation of timestamps (if > 2 txs).
        
        timestamps = []
        if in_edges:
            timestamps.extend([d['Timestamp'] for u, _, d in in_edges])
        if out_edges:
            timestamps.extend([d['Timestamp'] for _, v, d in out_edges])
            
        if len(timestamps) > 1:
            timestamps.sort()
            diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            avg_gap = np.mean(diffs)
            std_gap = np.std(diffs)
            node_feats['avg_time_gap'] = avg_gap
            node_feats['std_time_gap'] = std_gap
        else:
            node_feats['avg_time_gap'] = -1 # Indicator for no robust history
            node_feats['std_time_gap'] = -1

        features[node] = node_feats
        
    features_df = pd.DataFrame.from_dict(features, orient='index')
    features_df.index.name = 'Wallet_ID'
    features_df.reset_index(inplace=True)
    
    # Merge with Labels
    # Note: Not all wallets in the graph have labels.
    # We perform a left join on the features (which covers all graph nodes) 
    # and the provided labels. Unlabeled nodes will have NaN labels (to be used for inference).
    final_df = features_df.merge(labels_df, on='Wallet_ID', how='left')
    
    # Fill NaN labels with -1 or keep as NaN to distinguish "unknown" from "normal"
    # For training, we will filter for known labels (0 or 1).
    
    return final_df

if __name__ == "__main__":
    df_trans, df_labels = load_data()
    df_features = compute_features(df_trans, df_labels)
    print(df_features.head())
    print("Feature Engineering Complete.")
    # Optional: Save to CSV for inspection
    # df_features.to_csv("features_debug.csv", index=False)
