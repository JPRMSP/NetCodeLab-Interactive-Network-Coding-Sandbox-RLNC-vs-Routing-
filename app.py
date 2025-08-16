import streamlit as st
import networkx as nx
import numpy as np
from collections import defaultdict, deque
from pyvis.network import Network
import tempfile, os
import random
import matplotlib.pyplot as plt

# -------------------------------
# Utility: GF(2) linear algebra
# -------------------------------
def gf2_rank(A):
    """Return rank over GF(2) using row-reduction on a copy of A (numpy uint8/boolean)."""
    if A.size == 0:
        return 0
    A = (A.copy() & 1).astype(np.uint8)
    m, n = A.shape
    rank = 0
    col = 0
    for r in range(m):
        # find pivot in or below row r
        pivot = -1
        while col < n and pivot == -1:
            for i in range(r, m):
                if A[i, col]:
                    pivot = i
                    break
            if pivot == -1:
                col += 1
        if pivot == -1:
            break
        # swap rows
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        # eliminate below
        for i in range(r + 1, m):
            if A[i, col]:
                A[i, :] ^= A[r, :]
        rank += 1
        col += 1
    return rank

def gf2_rref(A, B=None):
    """
    Reduced row echelon form over GF(2).
    If B is provided (same number of rows), apply identical row ops to B (for decoding payloads).
    Returns (R, T) where R is RREF(A), and T is transformed B (or None).
    """
    A = (A.copy() & 1).astype(np.uint8)
    T = None if B is None else (B.copy() & 1).astype(np.uint8)
    m, n = A.shape
    row = 0
    for col in range(n):
        # find pivot
        pivot = -1
        for r in range(row, m):
            if A[r, col]:
                pivot = r
                break
        if pivot == -1:
            continue
        # swap to current row
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
            if T is not None:
                T[[row, pivot]] = T[[pivot, row]]
        # eliminate above and below
        for r in range(m):
            if r != row and A[r, col]:
                A[r, :] ^= A[row, :]
                if T is not None:
                    T[r, :] ^= T[row, :]
        row += 1
        if row == m:
            break
    return A, T

# -------------------------------
# Packets & Nodes
# -------------------------------
class Packet:
    def __init__(self, coeff, payload):
        # coeff: (k,) uint8 in {0,1}, payload: (L,) uint8 (bitwise XOR composition)
        self.coeff = (coeff & 1).astype(np.uint8)
        self.payload = payload.astype(np.uint8)

def combine_packets(packets):
    """Random linear combination over GF(2). Choose random 0/1 coefficients (at least one 1)."""
    if not packets:
        return None
    k = packets[0].coeff.shape[0]
    L = packets[0].payload.shape[0]
    # choose random subset
    mask = np.zeros(len(packets), dtype=np.uint8)
    # ensure at least one chosen
    chosen_indices = []
    for i in range(len(packets)):
        if random.random() < 0.5:
            mask[i] = 1
            chosen_indices.append(i)
    if not chosen_indices:
        idx = random.randrange(len(packets))
        mask[idx] = 1
        chosen_indices = [idx]
    coeff = np.zeros(k, dtype=np.uint8)
    payload = np.zeros(L, dtype=np.uint8)
    for i, p in enumerate(packets):
        if mask[i]:
            coeff ^= p.coeff
            payload ^= p.payload
    return Packet(coeff, payload)

def xor_merge(packets):
    """Simple XOR of all given packets (deterministic), used in XOR-only mode."""
    if not packets:
        return None
    k = packets[0].coeff.shape[0]
    L = packets[0].payload.shape[0]
    coeff = np.zeros(k, dtype=np.uint8)
    payload = np.zeros(L, dtype=np.uint8)
    for p in packets:
        coeff ^= p.coeff
        payload ^= p.payload
    return Packet(coeff, payload)

# -------------------------------
# Simulation Core
# -------------------------------
class NetCodeSim:
    def __init__(self):
        self.G = nx.DiGraph()  # edges have 'capacity' (packets per tick) and 'cost' (energy)
        self.source_nodes = set()
        self.sink_nodes = set()
        self.node_buffers = defaultdict(list)  # node -> list[Packet] available to send (from prior ticks)
        self.next_buffers = defaultdict(list)  # staging for next tick arrivals
        self.k = 2
        self.payload_len = 32
        self.t = 0
        self.loss_prob = 0.0
        self.energy_per_tx = 1.0
        self.total_tx = 0
        self.total_dropped = 0
        self.mode = "RLNC"  # "RLNC", "XOR", "Routing"
        self.systematic = True
        self.auto = False
        # Decoding state per sink
        self.sink_matrix = {}  # node -> (A, P) collected so far
        self.sink_rank_hist = defaultdict(list)  # node -> list of (t, rank)
        self.decoding_time = {}  # node -> first t where rank==k (or None)
        # Routing baseline path memory (per source->sink)
        self.routing_paths = {}  # (s, t) -> list of nodes

    def reset_state(self):
        self.node_buffers = defaultdict(list)
        self.next_buffers = defaultdict(list)
        self.t = 0
        self.total_tx = 0
        self.total_dropped = 0
        self.sink_matrix = {}
        self.sink_rank_hist = defaultdict(list)
        self.decoding_time = {}
        # Initialize sources with originals if systematic
        self._init_sources()

    def set_params(self, k, payload_len, loss_prob, energy_per_tx, mode, systematic=True):
        self.k = int(k)
        self.payload_len = int(payload_len)
        self.loss_prob = float(loss_prob)
        self.energy_per_tx = float(energy_per_tx)
        self.mode = mode
        self.systematic = systematic

    def add_node(self, node, kind="intermediate"):
        self.G.add_node(node, kind=kind)
        if kind == "source":
            self.source_nodes.add(node)
        if kind == "sink":
            self.sink_nodes.add(node)

    def add_edge(self, u, v, capacity=1, cost=1.0):
        self.G.add_edge(u, v, capacity=int(capacity), cost=float(cost))

    def _init_sources(self):
        # Create k original packets per source; if multiple sources, split k evenly (or replicate)
        if not self.source_nodes:
            return
        per_source = max(1, self.k // max(1, len(self.source_nodes)))
        remaining = self.k
        rng = np.random.default_rng(12345)
        for s in self.source_nodes:
            cnt = per_source if remaining >= per_source else remaining
            cnt = max(0, cnt)
            for i in range(cnt):
                idx = self.k - remaining
                coeff = np.zeros(self.k, dtype=np.uint8)
                if self.systematic and idx < self.k:
                    coeff[idx] = 1
                else:
                    # random coeffs for non-systematic (fallback)
                    coeff = (rng.integers(0, 2, size=self.k)).astype(np.uint8)
                    if not np.any(coeff):
                        coeff[rng.integers(0, self.k)] = 1
                payload = rng.integers(0, 256, size=self.payload_len, dtype=np.uint8)
                pkt = Packet(coeff, payload)
                self.node_buffers[s].append(pkt)
                remaining -= 1
                if remaining == 0:
                    break
            if remaining == 0:
                break
        # If k not covered (e.g., multi-source), inject additional random combinations at runtime.

    def compute_maxflow_bound(self):
        """For multicast, bound is min over sinks of max s->t across super-source (if multiple sources)."""
        if not self.source_nodes or not self.sink_nodes:
            return 0.0, {}
        # Build super-source
        H = nx.DiGraph()
        for u, v, data in self.G.edges(data=True):
            H.add_edge(u, v, capacity=data.get("capacity", 1))
        super_s = "__super__"
        for s in self.source_nodes:
            H.add_edge(super_s, s, capacity=10**6)
        per_sink = {}
        min_bound = float("inf")
        for t in self.sink_nodes:
            flow_val, _ = nx.maximum_flow(H, super_s, t, flow_func=nx.algorithms.flow.edmonds_karp)
            per_sink[t] = flow_val
            min_bound = min(min_bound, flow_val)
        return min_bound, per_sink

    def prepare_routing_paths(self):
        self.routing_paths = {}
        for s in self.source_nodes:
            for t in self.sink_nodes:
                try:
                    path = nx.shortest_path(self.G, s, t)
                    self.routing_paths[(s, t)] = path
                except nx.NetworkXNoPath:
                    pass

    def step(self):
        """Advance simulation by one tick."""
        self.t += 1
        # Source behavior: inject more combos after systematic phase if desired
        for s in self.source_nodes:
            if self.mode in ("RLNC", "XOR"):
                # push one new random combo per tick to keep traffic (can be tuned)
                if self.k > 0:
                    # If buffers empty at source, create another random combo
                    if not self.node_buffers[s]:
                        rng = np.random.default_rng(999 + self.t)
                        coeff = (rng.integers(0, 2, size=self.k)).astype(np.uint8)
                        if not np.any(coeff):
                            coeff[rng.integers(0, self.k)] = 1
                        payload = rng.integers(0, 256, size=self.payload_len, dtype=np.uint8)
                        self.node_buffers[s].append(Packet(coeff, payload))

        # Decide transmissions based on current buffers
        sent_edge_count = 0
        dropped = 0
        for u, v, data in self.G.edges(data=True):
            cap = data.get("capacity", 1)
            # choose packet to send from u for this edge, up to capacity
            for _ in range(cap):
                pkt = None
                if self.mode == "RLNC":
                    pkt = combine_packets(self.node_buffers[u]) if self.node_buffers[u] else None
                elif self.mode == "XOR":
                    # deterministic XOR of all available packets
                    pkt = xor_merge(self.node_buffers[u]) if self.node_buffers[u] else None
                elif self.mode == "Routing":
                    # forward first unseen packet along shortest paths toward v if v is next hop to any sink
                    pkt = self._pick_routing_packet(u, v)
                if pkt is None:
                    continue
                sent_edge_count += 1
                # stochastic loss
                if random.random() < self.loss_prob:
                    dropped += 1
                    continue
                # deliver at next tick
                self.next_buffers[v].append(pkt)

        # Move next_buffers to node_buffers (arrival)
        for node, lst in self.next_buffers.items():
            if lst:
                self.node_buffers[node].extend(lst)
        self.next_buffers = defaultdict(list)
        self.total_tx += sent_edge_count
        self.total_dropped += dropped

        # Decode at sinks: update rank hist and decoding times
        for tnode in self.sink_nodes:
            pkts = self.node_buffers[tnode]
            if not pkts:
                self.sink_rank_hist[tnode].append((self.t, 0))
                continue
            A = np.vstack([p.coeff for p in pkts]).astype(np.uint8)
            rank = gf2_rank(A)
            self.sink_rank_hist[tnode].append((self.t, rank))
            if tnode not in self.decoding_time and rank >= self.k:
                self.decoding_time[tnode] = self.t

    def _pick_routing_packet(self, u, v):
        """Simple routing: if edge (u->v) lies on any precomputed shortest path to a sink, forward queue head."""
        if not self.routing_paths:
            self.prepare_routing_paths()
        # If u->v appears as consecutive nodes on any (s,t) path and u is on that path before v:
        for (s, t), path in self.routing_paths.items():
            for i in range(len(path) - 1):
                if path[i] == u and path[i + 1] == v:
                    # forward first packet (no mixing)
                    if self.node_buffers[u]:
                        return self.node_buffers[u][0]
        return None

    # -------------------------------
    # Visualization helpers
    # -------------------------------
    def pyvis_html(self, height=460):
        net = Network(height=f"{height}px}", width="100%", directed=True, notebook=False)
        # Add nodes with color by kind
        for n, attrs in self.G.nodes(data=True):
            kind = attrs.get("kind", "intermediate")
            color = {"source": "#16a34a", "sink": "#ef4444", "intermediate": "#3b82f6"}.get(kind, "#3b82f6")
            title = f"{n} ({kind})"
            net.add_node(n, label=str(n), color=color, title=title)
        for u, v, data in self.G.edges(data=True):
            cap = data.get("capacity", 1)
            cost = data.get("cost", 1.0)
            net.add_edge(u, v, label=f"c={cap}, e={cost}", title=f"capacity={cap}, energy={cost}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.show(tmp.name)
            html = open(tmp.name, "r", encoding="utf-8").read()
        os.unlink(tmp.name)
        return html

    def metrics(self):
        # Throughput (innovative rate) approximation: min rank increase slope across sinks
        per_sink = {}
        for tnode in self.sink_nodes:
            series = self.sink_rank_hist.get(tnode, [])
            rank = series[-1][1] if series else 0
            per_sink[tnode] = {
                "rank": rank,
                "decoded_at": self.decoding_time.get(tnode, None)
            }
        energy = self.total_tx * self.energy_per_tx
        return per_sink, energy, self.total_tx, self.total_dropped


# -------------------------------
# Presets (Acyclic, Cyclic, Butterfly, Multicast)
# -------------------------------
def load_preset(sim, name):
    sim.G.clear()
    sim.source_nodes.clear()
    sim.sink_nodes.clear()
    if name == "Butterfly Multicast (classic)":
        # Nodes
        for n in ["S", "A", "B", "C", "D", "T1", "T2"]:
            kind = "intermediate"
            if n == "S":
                kind = "source"
            if n in ("T1", "T2"):
                kind = "sink"
            sim.add_node(n, kind=kind)
        # Edges (capacity=1)
        edges = [("S","A"), ("S","B"),
                 ("A","C"), ("B","C"),
                 ("A","T1"), ("B","T2"),
                 ("C","D"),
                 ("D","T1"), ("D","T2")]
        for u,v in edges:
            sim.add_edge(u,v,capacity=1,cost=1.0)
    elif name == "Acyclic Two-Path":
        for n in ["S","X","Y","T"]:
            sim.add_node(n, kind="intermediate")
        sim.G.nodes["S"]["kind"]="source"; sim.source_nodes.add("S")
        sim.G.nodes["T"]["kind"]="sink"; sim.sink_nodes.add("T")
        edges = [("S","X"),("S","Y"),("X","T"),("Y","T")]
        for u,v in edges:
            sim.add_edge(u,v,capacity=1,cost=1.0)
    elif name == "Cyclic Triangle":
        for n in ["S","U","V","T"]:
            kind="intermediate"
            sim.add_node(n, kind=kind)
        sim.G.nodes["S"]["kind"]="source"; sim.source_nodes.add("S")
        sim.G.nodes["T"]["kind"]="sink"; sim.sink_nodes.add("T")
        edges = [("S","U"),("U","V"),("V","S"),("U","T"),("V","T")]
        for u,v in edges:
            sim.add_edge(u,v,capacity=1,cost=1.0)
    elif name == "Multisource Multicast":
        # Two sources to two sinks with shared middle
        for n in ["S1","S2","M1","M2","T1","T2"]:
            kind="intermediate"
            if n in ("S1","S2"): kind="source"
            if n in ("T1","T2"): kind="sink"
            sim.add_node(n, kind=kind)
        edges = [("S1","M1"),("S2","M1"),("M1","M2"),("M2","T1"),("M2","T2"),
                 ("S1","T1"),("S2","T2")]
        for u,v in edges:
            sim.add_edge(u,v,capacity=1,cost=1.0)
    sim.reset_state()

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="NetCodeLab: Interactive Network Coding", layout="wide")
st.title("NetCodeLab — Interactive Network Coding (GF(2))")
st.caption("FI1938: Max-flow/Min-cut, RLNC vs Routing, decoding delay & energy — real-time, no datasets/models.")

# Session state
if "sim" not in st.session_state:
    st.session_state.sim = NetCodeSim()
sim: NetCodeSim = st.session_state.sim

# Sidebar controls
with st.sidebar:
    st.header("Setup")
    preset = st.selectbox("Topology preset", ["Butterfly Multicast (classic)", "Acyclic Two-Path", "Cyclic Triangle", "Multisource Multicast"])
    if st.button("Load Preset / Reset"):
        load_preset(sim, preset)
    st.divider()
    k = st.number_input("k (original packets)", min_value=1, max_value=16, value=2, step=1)
    payload_len = st.number_input("Payload length (bytes)", min_value=8, max_value=2048, value=32, step=8)
    loss_prob = st.slider("Loss probability per transmission", 0.0, 0.5, 0.05, 0.01)
    energy_per_tx = st.number_input("Energy per transmission (arbitrary units)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    mode = st.selectbox("Mode", ["RLNC", "XOR", "Routing"])
    systematic = st.checkbox("Systematic source (send identity first)", value=True)
    if st.button("Apply Parameters"):
        sim.set_params(k, payload_len, loss_prob, energy_per_tx, mode, systematic)
        sim.reset_state()

    st.divider()
    st.subheader("Manual editing")
    add_node_id = st.text_input("Add node ID")
    node_kind = st.selectbox("Node kind", ["intermediate","source","sink"])
    if st.button("Add Node"):
        if add_node_id.strip():
            sim.add_node(add_node_id.strip(), kind=node_kind)
            st.success(f"Added node {add_node_id} ({node_kind})")
    col1, col2 = st.columns(2)
    with col1:
        u = st.text_input("Edge from (u)")
    with col2:
        v = st.text_input("Edge to (v)")
    cap = st.number_input("Edge capacity (packets/tick)", min_value=1, max_value=8, value=1, step=1)
    cost = st.number_input("Edge energy cost", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    if st.button("Add Edge u→v"):
        if u.strip() and v.strip():
            sim.add_edge(u.strip(), v.strip(), capacity=cap, cost=cost)
            st.success(f"Added edge {u}→{v} (c={cap}, e={cost})")

# Main layout
left, right = st.columns([1.1, 1.2])

with left:
    st.subheader("Network")
    html = sim.pyvis_html(height=480)
    st.components.v1.html(html, height=500, scrolling=True)

    st.subheader("Theoretical Bounds")
    bound, per_sink = sim.compute_maxflow_bound()
    st.write(f"**Admissible multicast rate bound (min over sinks):** `{bound}` packets/tick")
    for tnode, f in per_sink.items():
        st.write(f"- Max flow to `{tnode}`: `{f}`")

    if sim.mode == "Routing":
        sim.prepare_routing_paths()
        if sim.routing_paths:
            st.caption("Routing shortest paths:")
            for (s,t), path in sim.routing_paths.items():
                st.write(f"{s} → {t}: {' → '.join(path)}")

with right:
    st.subheader("Simulation")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Step"):
        sim.step()
    if c2.button("Step ×10"):
        for _ in range(10):
            sim.step()
    if c3.button("Auto 20 ticks"):
        for _ in range(20):
            sim.step()
    if c4.button("Reset"):
        sim.reset_state()

    per_sink, energy, total_tx, total_drop = sim.metrics()
    st.metric("Time ticks", sim.t)
    st.metric("Total transmissions", total_tx)
    st.metric("Dropped (loss)", total_drop)
    st.metric("Energy used", f"{energy:.2f}")

    # Sinks table
    st.write("### Sink Status")
    rows = []
    for tnode, info in per_sink.items():
        rows.append({
            "Sink": tnode,
            "Current Rank": info["rank"],
            "Decoded at tick": info["decoded_at"] if info["decoded_at"] is not None else "-"
        })
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("No sinks configured.")

    # Plot rank over time per sink
    if sim.sink_rank_hist:
        st.write("### Rank Progress (innovative packets received)")
        for tnode, series in sim.sink_rank_hist.items():
            ts = [t for (t, _) in series]
            rs = [r for (_, r) in series]
            fig = plt.figure()
            plt.plot(ts, rs)
            plt.xlabel("Ticks")
            plt.ylabel("Rank")
            plt.title(f"Sink {tnode} — Rank vs Time")
            st.pyplot(fig)

st.divider()
with st.expander("What to say during viva (cheat sheet)"):
    st.markdown("""
- **Max-flow/Min-cut** bounds the achievable multicast rate; RLNC achieves the bound in acyclic networks (and with suitable scheduling also in cyclic).
- **Routing vs Coding**: Coding removes the butterfly bottleneck by mixing at the intermediate node; both sinks decode from innovative combinations.
- **Random Linear Network Coding (GF(2))**: Nodes send `aX ⊕ bY ⊕ ...`; sinks collect combinations until matrix rank = `k` then decode (here we track rank and decoding time).
- **Robustness**: With random losses, innovative rate remains high; decoding delay grows slowly.
- **Energy**: We count per-transmission energy; coding can reduce retransmissions in multicast compared to naive routing.
""")
