import streamlit as st
from pyvis.network import Network
import networkx as nx
import tempfile
import os

# ----------------------------
# Simulation Class
# ----------------------------
class Simulation:
    def __init__(self):
        self.G = nx.DiGraph()
        self.metrics_data = {}

    def add_nodes_and_edges(self):
        """Example network (you can replace with RLNC logic)."""
        self.G.add_nodes_from([1, 2, 3, 4])
        self.G.add_edges_from([(1, 2), (2, 3), (3, 4), (1, 4)])

    def run(self, method="Routing"):
        """Run either Routing or RLNC (placeholder)."""
        if method == "Routing":
            self.metrics_data = {
                "Method": "Routing",
                "Packets Sent": 10,
                "Packets Received": 9,
                "Loss": "10%"
            }
        else:
            self.metrics_data = {
                "Method": "RLNC",
                "Packets Sent": 10,
                "Packets Received": 10,
                "Loss": "0%"
            }

    def pyvis_html(self, height=480):
        """Render PyVis network and return HTML string."""
        net = Network(height=f"{height}px", width="100%", directed=True, notebook=False)
        net.from_nx(self.G)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.show(tmp.name)
            tmp_path = tmp.name

        with open(tmp_path, "r", encoding="utf-8") as f:
            html_str = f.read()

        os.remove(tmp_path)  # cleanup
        return html_str

    def metrics(self):
        return self.metrics_data


# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title="RLNC vs Routing Sandbox", layout="wide")
    st.title("🌐 Interactive Network Coding Sandbox (RLNC vs Routing)")

    sim = Simulation()
    sim.add_nodes_and_edges()

    # Sidebar controls
    st.sidebar.header("⚙️ Simulation Settings")
    method = st.sidebar.radio("Choose Method", ["Routing", "RLNC"])
    run_button = st.sidebar.button("Run Simulation")

    if run_button:
        sim.run(method)
        st.success(f"Simulation completed using {method}!")

        # Display metrics
        st.subheader("📊 Metrics")
        st.json(sim.metrics())

        # Display network graph
        st.subheader("🖧 Network Graph")
        from streamlit.components.v1 import html
        html(sim.pyvis_html(height=500), height=550)


if __name__ == "__main__":
    main()
