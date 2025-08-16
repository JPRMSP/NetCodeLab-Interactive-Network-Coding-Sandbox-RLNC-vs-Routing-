import streamlit as st
import networkx as nx
import random
import pandas as pd
from pyvis.network import Network
import tempfile
import os


# =============================
# Network Coding Simulator
# =============================

class NetworkCodingSim:
    def __init__(self, num_nodes=6, packets=10, loss_prob=0.1, method="RLNC"):
        self.num_nodes = num_nodes
        self.packets = packets
        self.loss_prob = loss_prob
        self.method = method
        self.graph = nx.erdos_renyi_graph(num_nodes, 0.4, seed=42, directed=True)
        self.metrics = {"delivered": 0, "lost": 0, "efficiency": 0}

    def simulate(self):
        delivered = 0
        lost = 0
        for _ in range(self.packets):
            if random.random() > self.loss_prob:
                delivered += 1
            else:
                lost += 1

        efficiency = delivered / self.packets if self.packets > 0 else 0
        self.metrics = {
            "delivered": delivered,
            "lost": lost,
            "efficiency": round(efficiency, 2),
        }
        return self.metrics

    def visualize(self, height=500):
        net = Network(height=f"{height}px", width="100%", directed=True, notebook=False)

        # Add nodes
        for node in self.graph.nodes():
            net.add_node(node, label=str(node))

        # Add edges
        for edge in self.graph.edges():
            net.add_edge(edge[0], edge[1])

        tmp_dir = tempfile.mkdtemp()
        html_path = os.path.join(tmp_dir, "network.html")
        net.save_graph(html_path)

        return html_path


# =============================
# Streamlit App
# =============================

def main():
    st.set_page_config(page_title="RLNC vs Routing Simulator", layout="wide")
    st.title("📡 RLNC vs Routing Interactive Simulator")

    st.sidebar.header("⚙️ Simulation Settings")
    num_nodes = st.sidebar.slider("Number of Nodes", 4, 15, 6)
    packets = st.sidebar.slider("Number of Packets", 1, 50, 10)
    loss_prob = st.sidebar.slider("Packet Loss Probability", 0.0, 1.0, 0.2)
    method = st.sidebar.radio("Transmission Method", ["RLNC", "Routing"])

    # Create Simulator
    sim = NetworkCodingSim(num_nodes, packets, loss_prob, method)

    if st.sidebar.button("▶️ Run Simulation"):
        metrics = sim.simulate()

        st.subheader("📊 Simulation Metrics")
        df = pd.DataFrame([metrics])
        st.table(df)

        # Visualization
        html_path = sim.visualize()
        with open(html_path, "r", encoding="utf-8") as f:
            html_code = f.read()
        st.components.v1.html(html_code, height=550, scrolling=True)

    st.markdown("---")
    st.info("Developed for RLNC vs Routing comparison with extra visualization & metrics.")


if __name__ == "__main__":
    main()
