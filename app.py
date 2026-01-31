
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import config
import feature_engineering
import graph_utils
import model
import os

# Set page config
st.set_page_config(layout="wide", page_title="AML Investigation System")

# Load Data and Model (Cached)
@st.cache_resource
def load_system():
    df_trans, df_labels = feature_engineering.load_data()
    
    # Calculate features once
    df_features = feature_engineering.compute_features(df_trans, df_labels)
    
    # Build full graph for querying
    G = feature_engineering.build_graph(df_trans)
    
    # Load Model
    clf, scaler = model.load_artifacts()
    
    return df_trans, df_features, G, clf, scaler

# Load everything
try:
    df_trans, df_features, G, clf, scaler = load_system()
except Exception as e:
    st.error(f"Error loading system: {e}. Please ensure model is trained (`python model.py`).")
    st.stop()

# Helper to get node features
def get_node_details(node_id):
    row = df_features[df_features['Wallet_ID'] == node_id]
    if not row.empty:
        return row.iloc[0].to_dict()
    return None

# Sidebar - Investigation Controls
st.sidebar.title("🕵️ AML Investigator")
st.sidebar.markdown("---")

# Search
if 'current_wallet' not in st.session_state:
    st.session_state.current_wallet = df_features['Wallet_ID'].iloc[0]

search_input = st.sidebar.text_input("Search Wallet ID", value=st.session_state.current_wallet)
if search_input:
    st.session_state.current_wallet = search_input

# Controls
st.sidebar.markdown("### Visualization Settings")
# INCREASED DEFAULT DEPTH TO 2
depth = st.sidebar.slider("Layer Depth", 1, 3, 2)

st.sidebar.markdown("---")

# Main Logic
center_node = st.session_state.current_wallet

if center_node not in G:
    st.error(f"Wallet {center_node} not found in transaction history.")
else:
    # 1. Get Details & Prediction
    details = get_node_details(center_node)
    
    # Predict Risk
    risk_score = 0.0
    risk_label = "Unknown"
    risk_color = "gray"
    
    if details and clf:
        try:
            risk_score = model.predict_risk(clf, scaler, details)[0]
            if risk_score > 0.7:
                risk_label = "HIGH RISK"
                risk_color = "red"
            elif risk_score > 0.3:
                risk_label = "MEDIUM RISK"
                risk_color = "orange"
            else:
                risk_label = "LOW RISK"
                risk_color = "green"
        except Exception as e:
            st.warning(f"Could not predict risk: {e}")
    
    # 2. Main Dashboard Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"Transaction Graph: {center_node}")
        
        try:
            # Subgraph extraction
            subgraph = graph_utils.get_k_hop_subgraph(G, center_node, k=depth)
            
            if subgraph:
                # Convert NetworkX graph to agraph Nodes/Edges
                nodes = []
                edges = []
                
                # Keep track of added nodes to avoid duplicates
                added_nodes = set()
                
                for n_id in subgraph.nodes():
                    if n_id in added_nodes:
                        continue
                    
                    added_nodes.add(n_id)
                    
                    # Determine color/shape based on center
                    color = "#97C2FC" # Default blue
                    size = 25
                    label = str(n_id)
                    
                    if n_id == center_node:
                        color = "#FF0000" if risk_label == "HIGH RISK" else "#0000FF"
                        size = 40
                    
                    nodes.append(Node(id=n_id, label=label, size=size, color=color))
                
                for u, v, data in subgraph.edges(data=True):
                    # Edge tooltip
                    amt = data.get('Amount', 0)
                    ts = str(data.get('Timestamp', ''))
                    title = f"Amt: {amt:.2f}\nTime: {ts}"
                    edges.append(Edge(source=u, target=v, title=title))
                
                # Agraph Config
                config_ag = Config(width=1000, 
                                height=600, 
                                directed=True,
                                nodeHighlightBehavior=True, 
                                highlightColor="#F7A7A6", 
                                collapsible=False,
                                physics={
                                    "enabled": True,
                                    "stabilization": {
                                        "enabled": True,
                                        "iterations": 1000 # Pre-stabilize to avoid exploding graph
                                    },
                                    "barnesHut": {
                                        "gravitationalConstant": -5000, #Stronger repulsion
                                        "centralGravity": 0.1,
                                        "springLength": 100
                                    }
                                })
                
                # Render Graph
                return_value = agraph(nodes=nodes, edges=edges, config=config_ag)
                
                # INTERACTIVITY: If node clicked, update session state
                if return_value and return_value != center_node:
                    st.session_state.current_wallet = return_value
                    st.rerun()
                    
            else:
                st.warning("No subgraph found.")
                
        except Exception as e:
            st.error(f"Error rendering graph: {e}")

    with col2:
        st.markdown(f"## Risk Analysis")
        st.markdown(f"<h1 style='color: {risk_color};'>{risk_label}</h1>", unsafe_allow_html=True)
        st.markdown(f"**Suspicion Score:** {risk_score:.4f}")
        
        if details:
            st.markdown("### Wallet DNA")
            st.metric("Fan In", details.get('fan_in'))
            st.metric("Fan Out", details.get('fan_out'))
            st.metric("Total Received", f"{details.get('total_in'):.2f}")
            st.metric("Total Sent", f"{details.get('total_out'):.2f}")
            st.metric("Mule Score", f"{details.get('mule_score'):.2f}")
            
            st.markdown("### Explanation")
            if details.get('fan_in') > 10 and details.get('fan_out') > 10:
                st.warning("High Fan-in/out indicates possible mixing/layering.")
            if risk_score > 0.8:
                st.error("Model flags this wallet as highly suspicious based on transactional patterns.")
            if details.get('balance_ratio') > 0.95 and details.get('balance_ratio') < 1.05 and details.get('total_in') > 10:
                 st.info("Pass-through behavior detected (In ≈ Out).")

        st.markdown("---")
        st.markdown("### Path Tracer")
        target_wallet = st.text_input("Trace to Target Wallet ID")
        if target_wallet and st.button("Find Path"):
            paths = graph_utils.trace_path(G, center_node, target_wallet)
            if paths:
                st.success(f"Found {len(paths)} paths!")
                for i, path in enumerate(paths[:3]):
                    st.write(f"**Path {i+1}:**")
                    path_str = " → ".join(path)
                    st.code(path_str)
            else:
                st.warning("No path found (max depth 5).")
