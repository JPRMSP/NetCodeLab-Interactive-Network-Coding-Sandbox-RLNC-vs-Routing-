import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import json

# ------------------------------
# Simulation Class
# ------------------------------
class Simulation:
    def __init__(self, method):
        self.method = method
        self.packets_sent = 10
        self.packets_received = 10 if method == "RLNC" else 8

    def run(self):
        # Placeholder: extend with real logic
        return True

    def metrics(self):
        loss = (
            0
            if self.packets_sent == self.packets_received
            else (self.packets_sent - self.packets_received) / self.packets_sent * 100
        )
        return {
            "Method": self.method,
            "Packets Sent": self.packets_sent,
            "Packets Received": self.packets_received,
            "Loss": f"{loss:.0f}%",
        }

    def pyvis_html(self, height=500):
        net = Network(height=f"{height}px", width="100%", directed=True, notebook=False)

        # Example network structure
        net.add_node(1, label="Source")
        net.add_node(2, label="Relay 1")
        net.add_node(3, label="Relay 2")
        net.add_node(4, label="Destination")

        net.add_edge(1, 2)
        net.add_edge(1, 3)
        net.add_edge(2, 4)
        net.add_edge(3, 4)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            html_content = open(tmp.name, "r", encoding="utf-8").read()
        return html_content


# ------------------------------
# Main Streamlit App
# ------------------------------
def main():
    st.set_page_config(page_title="Interactive Network Coding Sandbox", layout="wide")

    st.title("🌐 Interactive Network Coding Sandbox (RLNC vs Routing)")

    st.sidebar.header("⚙️ Simulation Settings")
    method = st.sidebar.radio("Choose Method", ["Routing", "RLNC"])

    if st.sidebar.button("Run Simulation"):
        sim = Simulation(method)
        sim.run()

        st.success(f"✅ Simulation completed using {method}!")

        # Show metrics
        st.subheader("📊 Metrics")
        metrics = sim.metrics()
        st.json(metrics)

        # Show network graph
        st.subheader("🖧 Network Graph")
        html_code = sim.pyvis_html(height=500)
        components.html(html_code, height=550, scrolling=True)


if __name__ == "__main__":
    main()
