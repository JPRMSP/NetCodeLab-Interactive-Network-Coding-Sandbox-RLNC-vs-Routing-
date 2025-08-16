import streamlit as st import networkx as nx import random import json import pandas as pd from pyvis.network import Network import tempfile

-------------------------------

Simulation Class

-------------------------------

class NetworkCodingSim: def init(self, num_nodes=6, packets=10, loss_prob=0.1, method="RLNC"): self.num_nodes = num_nodes self.packets = packets self.loss_prob = loss_prob self.method = method self.graph = nx.erdos_renyi_graph(num_nodes, 0.4, seed=42) self.metrics = {}

def run(self):
    sent, received = self.packets, 0
    for _ in range(self.packets):
        if random.random() > self.loss_prob:
            received += 1
    loss = (sent - received) / sent * 100
    self.metrics = {
        "Method": self.method,
        "Packets Sent": sent,
        "Packets Received": received,
        "Loss": f"{loss:.1f}%"
    }
    return self.metrics

def pyvis_html(self, height=500):
    net = Network(height=f"{height}px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(self.graph)

    # Add legend
    net.add_node("Source", color="green", shape="star")
    net.add_node("Destination", color="red", shape="triangle")

    # Save visualization
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.show(tmp.name)
        return tmp.name

-------------------------------

Streamlit UI

-------------------------------

def main(): st.set_page_config(page_title="Interactive Network Coding Sandbox", layout="wide")

st.title("🌐 Interactive Network Coding Sandbox (RLNC vs Routing)")
st.markdown("Compare Routing and RLNC under different network conditions.")

# Sidebar controls
st.sidebar.header("⚙️ Simulation Settings")
method = st.sidebar.radio("Choose Method", ["Routing", "RLNC", "Compare Both"])
num_nodes = st.sidebar.slider("Number of Nodes", 4, 12, 6)
packets = st.sidebar.slider("Packets Sent", 5, 50, 10)
loss_prob = st.sidebar.slider("Packet Loss Probability", 0.0, 0.5, 0.1, step=0.05)

run_btn = st.sidebar.button("▶️ Run Simulation")

# History tracker
if "history" not in st.session_state:
    st.session_state.history = []

if run_btn:
    if method == "Compare Both":
        results = []
        for m in ["Routing", "RLNC"]:
            sim = NetworkCodingSim(num_nodes=num_nodes, packets=packets, loss_prob=loss_prob, method=m)
            metrics = sim.run()
            results.append(metrics)
            graph_path = sim.pyvis_html()
            with open(graph_path, "r") as f:
                html_code = f.read()
            st.subheader(f"🖧 Network Graph - {m}")
            st.components.v1.html(html_code, height=550, scrolling=True)

        st.subheader("📊 Comparison Results")
        df = pd.DataFrame(results)
        st.dataframe(df)
        st.session_state.history.append(df)

    else:
        sim = NetworkCodingSim(num_nodes=num_nodes, packets=packets, loss_prob=loss_prob, method=method)
        metrics = sim.run()

        st.subheader(f"Simulation completed using {method}!")
        st.json(metrics)

        # Show network visualization
        graph_path = sim.pyvis_html()
        with open(graph_path, "r") as f:
            html_code = f.read()
        st.subheader("🖧 Network Graph")
        st.components.v1.html(html_code, height=550, scrolling=True)

        st.session_state.history.append(metrics)

# History Panel
if st.session_state.history:
    st.sidebar.subheader("📜 Simulation History")
    for i, h in enumerate(st.session_state.history[::-1], 1):
        st.sidebar.write(f"Run {i}:")
        st.sidebar.json(h)

# Export Results
if st.session_state.history:
    export_df = pd.DataFrame(st.session_state.history)
    st.download_button("⬇️ Download Results (CSV)", data=export_df.to_csv(index=False), file_name="sim_results.csv", mime="text/csv")
    st.download_button("⬇️ Download Results (JSON)", data=json.dumps(st.session_state.history, indent=2), file_name="sim_results.json", mime="application/json")

if name == "main": main()
