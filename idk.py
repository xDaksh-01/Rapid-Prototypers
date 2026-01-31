import pandas as pd
import networkx as nx
from pyvis.network import Network

# =========================
# 1. LOAD DATA (ONLY 1000 TX)
# =========================
tx = pd.read_csv("massive_transactions.csv")
tx = tx.head(300)

labels = pd.read_csv("massive_labels.csv")
illicit_wallets = set(labels[labels["Label"] == 1]["Wallet_ID"])

# =========================
# 2. BUILD GRAPH
# =========================
G = nx.DiGraph()

for _, row in tx.iterrows():
    G.add_edge(
        row["Source_Wallet_ID"],
        row["Dest_Wallet_ID"],
        amount=row["Amount"]
    )

# =========================
# 3. SIMPLE SUSPICION SCORE
# =========================
scores = {}

for n in G.nodes():
    fan_in = G.in_degree(n)
    fan_out = G.out_degree(n)

    neighbors = list(G.predecessors(n)) + list(G.successors(n))
    illicit_links = sum(1 for x in neighbors if x in illicit_wallets)

    score = 0.5 * illicit_links + 0.3 * fan_out + 0.2 * fan_in
    scores[n] = score

# Normalize
mx = max(scores.values()) if scores else 1
scores = {k: v / mx for k, v in scores.items()}

# =========================
# 4. COLOR FUNCTION
# =========================
def color(score):
    if score >= 0.7:
        return "#ef4444"   # red
    elif score >= 0.4:
        return "#facc15"   # yellow
    else:
        return "#22c55e"   # green

# =========================
# 5. CREATE BEAUTIFUL NETWORK
# =========================
net = Network(
    height="750px",
    width="100%",
    bgcolor="#f8fafc",
    font_color="#0f172a",
    directed=True,
    notebook=False
)

# ❌ No physics → clean static look
# Allow nodes to spread out once
net.repulsion(
    node_distance=180,
    central_gravity=0.25,
    spring_length=180,
    spring_strength=0.05,
    damping=0.9
)

net.toggle_physics(True)


# =========================
# 6. ADD NODES
# =========================
for n in G.nodes():
    s = scores[n]

    net.add_node(
        n,
        label=n,
        size=22 + s * 28,
        color=color(s),
        title=f"""
        Wallet: {n}<br>
        Suspicion Score: {round(s,2)}<br>
        Fan-in: {G.in_degree(n)}<br>
        Fan-out: {G.out_degree(n)}
        """
    )

# =========================
# 7. ADD EDGES
# =========================
for u, v, d in G.edges(data=True):
    net.add_edge(
        u, v,
        title=f"Amount: {d['amount']}",
        width=1
    )

# =========================
# 8. ADD LEGEND (HTML MAGIC)
# =========================
legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
background: white;
padding: 14px;
border-radius: 10px;
box-shadow: 0 4px 12px rgba(0,0,0,0.15);
font-family: Arial;
font-size: 14px;
">
<b>Legend</b><br><br>
<span style="color:#ef4444;">●</span> High Risk (Red)<br>
<span style="color:#facc15;">●</span> Medium Risk (Yellow)<br>
<span style="color:#22c55e;">●</span> Low Risk (Green)<br>
<br>
Arrow → Transaction Flow
</div>
"""

net.html += legend_html
# Freeze layout to avoid chaos

net.toggle_physics(False)

# =========================
# 9. EXPORT
# =========================
net.write_html("beautiful_laundering_graph.html", open_browser=True)

print("Saved: beautiful_laundering_graph.html")
