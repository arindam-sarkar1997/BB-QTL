import numpy as np
import pandas as pd
import networkx as nx

from scipy.stats import hypergeom, rankdata


def _bh_fdr(pvalues):
    """Benjamini-Hochberg FDR correction."""
    pvalues = np.asarray(pvalues, dtype=float)

    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]

    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out = np.empty(n)
    out[order] = q

    return out


def pathway_hub_subnetwork(
    df,
    node1="pwy1",
    node2="pwy2",
    weight="vc_mean",

    # -----------------------------
    # Parameters controlling hubs
    # -----------------------------
    strong_edge_quantile=0.90,
    hub_metric="hybrid",
    n_hubs=8,
    hub_fdr=None,
    hybrid_enrichment_weight=0.65,

    # ------------------------------------
    # Parameters controlling neighborhood
    # ------------------------------------
    neighbor_edge_quantile=0.95,
    top_k_per_hub=6,
    min_relative_weight=0.0,
    mutual_top_k=False,

    # ------------------------------------
    # Structure of returned subnetwork
    # ------------------------------------
    include_hub_hub_edges=True,
    include_neighbor_neighbor_edges=False,
    max_nodes=None,
):
    """
    Identify pathways enriched for strong interactions and construct a
    hub-centered weighted subnetwork.

    Parameters
    ----------
    df : pd.DataFrame
        Edge table.

    node1, node2 : str
        Columns containing pathway names.

    weight : str
        Interaction strength. Default = 'vc_mean'.

    strong_edge_quantile : float
        Global edge-weight quantile used to define a "strong interaction".
        Example:
            0.90 -> top 10% of interactions
            0.95 -> top 5%

    hub_metric : {'enrichment', 'strength', 'hybrid', 'pagerank'}
        Metric used to rank hubs.

        enrichment:
            pathways having unusually many strong interactions.

        strength:
            sum of all interaction weights.

        hybrid:
            weighted combination of enrichment and total strength.

        pagerank:
            weighted PageRank on the complete weighted graph.

    n_hubs : int
        Number of hub pathways.

    hub_fdr : float or None
        Optional FDR cutoff for enrichment before selecting top hubs.
        Example: hub_fdr=0.05.

    hybrid_enrichment_weight : float
        Relative contribution of enrichment to hybrid score.
        0.65 means:
            65% strong-edge enrichment
            35% total weighted strength

    neighbor_edge_quantile : float
        Minimum global edge quantile for adding non-hub neighbors.
        Usually should be >= strong_edge_quantile.

    top_k_per_hub : int
        Maximum number of strongest neighbors retained for each hub.

    min_relative_weight : float
        Additional hub-specific cutoff expressed relative to the strongest
        edge of that hub.

        Example:
            0.5 -> edge must be at least 50% as strong as hub's strongest edge.

    mutual_top_k : bool
        If True, h--v is retained only if:
            v is one of h's strongest neighbors
        AND
            h is one of v's strongest neighbors.

        This generates much more conservative subnetworks.

    include_hub_hub_edges : bool
        Include interactions between selected hubs.

    include_neighbor_neighbor_edges : bool
        Include strong interactions between non-hub nodes in the subnetwork.

    max_nodes : int or None
        Hard maximum on number of nodes.

    Returns
    -------
    G : networkx.Graph
        Weighted subnetwork.

    scores : pd.DataFrame
        Hub statistics for every pathway.

    edge_table : pd.DataFrame
        Edges included in G.

    info : dict
        Thresholds and selected hubs.
    """

    # ============================================================
    # Clean edge table
    # ============================================================

    x = df[[node1, node2, weight]].dropna().copy()

    # remove self interactions
    x = x[x[node1] != x[node2]]

    # ensure A-B and B-A are treated as same undirected edge
    x["_a"] = x[[node1, node2]].min(axis=1)
    x["_b"] = x[[node1, node2]].max(axis=1)

    # average duplicates if present
    x = (
        x.groupby(["_a", "_b"], as_index=False)[weight]
        .mean()
        .rename(columns={"_a": node1, "_b": node2})
    )

    nodes = pd.Index(
        sorted(set(x[node1]) | set(x[node2]))
    )

    # ============================================================
    # Define globally strong interactions
    # ============================================================

    strong_threshold = x[weight].quantile(strong_edge_quantile)

    is_strong = x[weight] >= strong_threshold

    total_edges = len(x)
    total_strong_edges = int(is_strong.sum())

    # ============================================================
    # Calculate node-level hub statistics
    # ============================================================

    rows = []

    for node in nodes:

        incident = x[
            (x[node1] == node) |
            (x[node2] == node)
        ]

        n_edges = len(incident)

        n_strong = int(
            (incident[weight] >= strong_threshold).sum()
        )

        # --------------------------------------------------------
        # Hypergeometric enrichment:
        #
        # Does this node contain more globally strong edges
        # than expected by chance?
        # --------------------------------------------------------

        enrichment_p = hypergeom.sf(
            n_strong - 1,
            total_edges,
            total_strong_edges,
            n_edges,
        )

        strength = incident[weight].sum()

        mean_weight = incident[weight].mean()

        top5_mean = (
            incident
            .nlargest(min(5, len(incident)), weight)[weight]
            .mean()
        )

        rows.append({
            "pathway": node,
            "n_edges": n_edges,
            "n_strong": n_strong,
            "enrichment_p": enrichment_p,
            "strength": strength,
            "mean_weight": mean_weight,
            "top5_mean": top5_mean,
        })

    scores = pd.DataFrame(rows)

    scores["enrichment_q"] = _bh_fdr(
        scores["enrichment_p"]
    )

    scores["enrichment_score"] = -np.log10(
        np.clip(
            scores["enrichment_p"],
            1e-300,
            1
        )
    )

    # percentile rankings make metrics comparable
    scores["strength_pct"] = (
        rankdata(scores["strength"], method="average")
        / len(scores)
    )

    scores["enrichment_pct"] = (
        rankdata(
            scores["enrichment_score"],
            method="average"
        )
        / len(scores)
    )

    scores["hybrid_score"] = (
        hybrid_enrichment_weight
        * scores["enrichment_pct"]

        +

        (1 - hybrid_enrichment_weight)
        * scores["strength_pct"]
    )

    # ============================================================
    # Weighted PageRank
    # ============================================================

    G_full = nx.Graph()

    G_full.add_weighted_edges_from(
        x[[node1, node2, weight]]
        .itertuples(index=False, name=None)
    )

    pagerank = nx.pagerank(
        G_full,
        weight="weight"
    )

    scores["pagerank"] = (
        scores["pathway"]
        .map(pagerank)
    )

    scores["pagerank_pct"] = (
        rankdata(
            scores["pagerank"],
            method="average"
        )
        / len(scores)
    )

    # ============================================================
    # Choose hubs
    # ============================================================

    metric_map = {
        "enrichment": "enrichment_score",
        "strength": "strength",
        "hybrid": "hybrid_score",
        "pagerank": "pagerank",
    }

    if hub_metric not in metric_map:
        raise ValueError(
            "hub_metric must be one of "
            "'enrichment', 'strength', "
            "'hybrid', or 'pagerank'"
        )

    metric = metric_map[hub_metric]

    candidates = scores.copy()

    if hub_fdr is not None:
        candidates = candidates[
            candidates["enrichment_q"] <= hub_fdr
        ]

    hubs = (
        candidates
        .nlargest(n_hubs, metric)
        ["pathway"]
        .tolist()
    )

    hub_set = set(hubs)

    # ============================================================
    # Select strong neighbors around hubs
    # ============================================================

    neighbor_threshold = (
        x[weight]
        .quantile(neighbor_edge_quantile)
    )

    hub_neighbors = {}

    for hub in hubs:

        incident = x[
            (x[node1] == hub) |
            (x[node2] == hub)
        ].copy()

        incident["neighbor"] = np.where(
            incident[node1] == hub,
            incident[node2],
            incident[node1],
        )

        # global cutoff + hub-relative cutoff
        cutoff = max(
            neighbor_threshold,
            incident[weight].max()
            * min_relative_weight
        )

        incident = (
            incident[
                incident[weight] >= cutoff
            ]
            .nlargest(
                top_k_per_hub,
                weight
            )
        )

        hub_neighbors[hub] = set(
            incident["neighbor"]
        )

    # ============================================================
    # Optional reciprocal / mutual top-k criterion
    # ============================================================

    if mutual_top_k:

        all_top_neighbors = {}

        for node in nodes:

            incident = x[
                (x[node1] == node) |
                (x[node2] == node)
            ].copy()

            incident["neighbor"] = np.where(
                incident[node1] == node,
                incident[node2],
                incident[node1],
            )

            cutoff = max(
                neighbor_threshold,
                incident[weight].max()
                * min_relative_weight
            )

            all_top_neighbors[node] = set(
                incident[
                    incident[weight] >= cutoff
                ]
                .nlargest(
                    top_k_per_hub,
                    weight
                )
                ["neighbor"]
            )

        for hub in hubs:

            hub_neighbors[hub] = {
                node
                for node in hub_neighbors[hub]
                if hub in all_top_neighbors[node]
            }

    # ============================================================
    # Collect selected nodes
    # ============================================================

    selected_nodes = set(hubs)

    for neighbor_set in hub_neighbors.values():
        selected_nodes |= neighbor_set

    # ============================================================
    # Optional global node cap
    # ============================================================

    if (
        max_nodes is not None
        and len(selected_nodes) > max_nodes
    ):

        non_hubs = list(
            selected_nodes - hub_set
        )

        max_hub_weight = {}

        for node in non_hubs:

            incident = x[
                (
                    (x[node1] == node)
                    & x[node2].isin(hubs)
                )
                |
                (
                    (x[node2] == node)
                    & x[node1].isin(hubs)
                )
            ]

            if len(incident):
                max_hub_weight[node] = (
                    incident[weight].max()
                )
            else:
                max_hub_weight[node] = -np.inf

        n_extra = max(
            0,
            max_nodes - len(hubs)
        )

        best_neighbors = sorted(
            non_hubs,
            key=max_hub_weight.get,
            reverse=True,
        )[:n_extra]

        selected_nodes = (
            hub_set |
            set(best_neighbors)
        )

    # ============================================================
    # Construct edge table
    # ============================================================

    edge_table = x[
        x[node1].isin(selected_nodes)
        &
        x[node2].isin(selected_nodes)
    ].copy()

    def retain_edge(row):

        a = row[node1]
        b = row[node2]

        a_hub = a in hub_set
        b_hub = b in hub_set

        # hub -- hub
        if a_hub and b_hub:
            return include_hub_hub_edges

        # hub -- neighbor
        if a_hub != b_hub:

            hub = a if a_hub else b
            neighbor = b if a_hub else a

            return (
                neighbor
                in hub_neighbors.get(
                    hub,
                    set()
                )
            )

        # neighbor -- neighbor
        return (
            include_neighbor_neighbor_edges
            and
            row[weight] >= neighbor_threshold
        )

    edge_table = edge_table[
        edge_table.apply(
            retain_edge,
            axis=1
        )
    ]

    edge_table = (
        edge_table
        .sort_values(
            weight,
            ascending=False
        )
        .reset_index(drop=True)
    )

    # ============================================================
    # Build final network
    # ============================================================

    G = nx.from_pandas_edgelist(
        edge_table,
        source=node1,
        target=node2,
        edge_attr=weight,
        create_using=nx.Graph(),
    )

    # ensure isolated hubs are retained
    G.add_nodes_from(hubs)

    nx.set_node_attributes(
        G,
        {
            node: node in hub_set
            for node in G.nodes
        },
        "is_hub",
    )

    score_lookup = (
        scores
        .set_index("pathway")
    )

    for attr in [
        "n_strong",
        "enrichment_score",
        "enrichment_q",
        "strength",
        "hybrid_score",
        "pagerank",
    ]:

        nx.set_node_attributes(
            G,
            score_lookup[attr]
            .to_dict(),
            attr,
        )

    # rank table according to chosen metric
    scores = (
        scores
        .sort_values(
            metric,
            ascending=False
        )
        .reset_index(drop=True)
    )

    info = {
        "strong_edge_threshold": strong_threshold,
        "neighbor_edge_threshold": neighbor_threshold,
        "hub_metric": hub_metric,
        "hubs": hubs,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
    }

    return G, scores, edge_table, info



import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_pathway_subnetwork(
    G,
    weight="vc_mean",
    hub_attr="is_hub",
    hub_score_attr="hybrid_score",
    figsize=(14, 11),
    layout="spring",
    seed=7,
    k=0.75,
    hub_size=(1000, 2200),
    neighbor_size=450,
    edge_width=(0.6, 4.5),
    edge_alpha=0.55,
    node_alpha=0.95,
    font_size=8,
    show_labels=True,
    title=None,
    ax=None,
):
    """
    Plot a hub-centered pathway interaction network.

    Parameters
    ----------
    G : networkx.Graph
        Graph returned by pathway_hub_subnetwork().

    weight : str
        Edge attribute containing interaction strength.
        Default = 'vc_mean'.

    hub_attr : str
        Boolean node attribute indicating whether a node is a hub.

    hub_score_attr : str
        Node attribute used to scale hub sizes.
        Default = 'hybrid_score'.

    figsize : tuple
        Figure size.

    layout : {'spring', 'kamada_kawai', 'circular', 'shell'}
        Graph layout.

    seed : int
        Random seed for spring layout.

    k : float
        Spacing parameter for spring layout.
        Larger values generally spread nodes further apart.

    hub_size : tuple
        Minimum and maximum hub node size.

    neighbor_size : float
        Size of non-hub nodes.

    edge_width : tuple
        Minimum and maximum edge width.

    edge_alpha : float
        Transparency of edges.

    node_alpha : float
        Transparency of nodes.

    font_size : float
        Label font size.

    show_labels : bool
        Whether to draw node labels.

    title : str or None
        Plot title.

    ax : matplotlib axis or None
        Existing axis to plot into.

    Returns
    -------
    fig, ax, pos
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # --------------------------------------------------
    # Identify hubs
    # --------------------------------------------------

    hubs = [
        n for n, d in G.nodes(data=True)
        if d.get(hub_attr, False)
    ]

    neighbors = [
        n for n in G.nodes
        if n not in hubs
    ]

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    if layout == "spring":

        pos = nx.spring_layout(
            G,
            seed=seed,
            weight=weight,
            k=k,
        )

    elif layout == "kamada_kawai":

        pos = nx.kamada_kawai_layout(
            G,
            weight=weight,
        )

    elif layout == "circular":

        pos = nx.circular_layout(G)

    elif layout == "shell":

        pos = nx.shell_layout(
            G,
            nlist=[hubs, neighbors]
        )

    else:
        raise ValueError(
            "layout must be one of: "
            "'spring', 'kamada_kawai', "
            "'circular', 'shell'"
        )

    # --------------------------------------------------
    # Scale edge widths using interaction strength
    # --------------------------------------------------

    edge_values = np.array([
        G[u][v].get(weight, 1.0)
        for u, v in G.edges()
    ], dtype=float)

    if len(edge_values) > 0:

        if edge_values.max() > edge_values.min():

            widths = (
                edge_width[0]
                +
                (edge_width[1] - edge_width[0])
                *
                (
                    edge_values - edge_values.min()
                )
                /
                (
                    edge_values.max() - edge_values.min()
                )
            )

        else:
            widths = np.full(
                len(edge_values),
                np.mean(edge_width)
            )

    else:
        widths = []

    # --------------------------------------------------
    # Scale hub node sizes using hub score
    # --------------------------------------------------

    hub_scores = np.array([
        G.nodes[n].get(hub_score_attr, 1.0)
        for n in hubs
    ], dtype=float)

    if len(hub_scores) > 0:

        if hub_scores.max() > hub_scores.min():

            hub_sizes = (
                hub_size[0]
                +
                (hub_size[1] - hub_size[0])
                *
                (
                    hub_scores - hub_scores.min()
                )
                /
                (
                    hub_scores.max() - hub_scores.min()
                )
            )

        else:
            hub_sizes = np.full(
                len(hubs),
                np.mean(hub_size)
            )

    else:
        hub_sizes = []

    # --------------------------------------------------
    # Draw edges
    # --------------------------------------------------

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        alpha=edge_alpha,
        ax=ax,
    )

    # --------------------------------------------------
    # Draw non-hubs
    # --------------------------------------------------

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=neighbors,
        node_size=neighbor_size,
        alpha=node_alpha,
        ax=ax,
    )

    # --------------------------------------------------
    # Draw hubs as squares
    # --------------------------------------------------

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=hubs,
        node_size=hub_sizes,
        node_shape="s",
        alpha=node_alpha,
        ax=ax,
    )

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------

    if show_labels:

        nx.draw_networkx_labels(
            G,
            pos,
            font_size=font_size,
            ax=ax,
        )

    # --------------------------------------------------
    # Title
    # --------------------------------------------------

    if title is None:

        title = (
            f"Pathway interaction subnetwork\n"
            f"{G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )

    ax.set_title(title)

    ax.axis("off")

    fig.tight_layout()

    return fig, ax, pos



import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def plot_top_interaction_network(
    df,
    k=10,
    source="pwy1",
    target="pwy2",
    weight="vc_mean",
    n_hubs=2,
    hub_metric="degree",
    spring_k=0.15,
    iterations=500,
    n_layout_trials=50,
    hub_center_strength=2.0,
    figsize=(8, 8),
    font_size=10,
    min_edge_width=0.5,
    max_edge_width=4.5,
    edge_alpha=0.6,
    seed=1,
):
    """
    Plot the top-k pathway interactions while:
      1. placing hubs preferentially near the center
      2. searching for a spring layout with fewer edge crossings

    Parameters
    ----------
    df : pd.DataFrame
        Edge dataframe.

    k : int
        Number of strongest interactions to retain.

    source, target : str
        Columns containing pathway names.

    weight : str
        Edge-weight column.

    n_hubs : int
        Number of hub nodes to favor toward the center.

    hub_metric : {'degree', 'weighted_degree'}
        How hubs are identified.

    spring_k : float
        NetworkX spring-layout spacing.
        Smaller -> more compact.

    iterations : int
        Spring-layout optimization iterations.

    n_layout_trials : int
        Number of random initializations to try.
        More trials generally gives fewer crossings.

    hub_center_strength : float
        How strongly to favor layouts with hubs near the origin.

        0:
            optimize crossings only

        larger values:
            increasingly prefer central hubs

    figsize : tuple
        Figure size.

    font_size : float
        Node-label size.

    min_edge_width, max_edge_width : float
        Edge-width range.

    edge_alpha : float
        Edge transparency.

    seed : int
        Base random seed.

    Returns
    -------
    G : nx.Graph
        Graph of top-k interactions.

    pos : dict
        Selected node positions.

    hubs : list
        Selected hub nodes.

    top_edges : pd.DataFrame
        Top-k dataframe used to construct the graph.
    """

    # --------------------------------------------------
    # Top k strongest interactions
    # --------------------------------------------------

    top_edges = df.nlargest(k, weight).copy()

    # --------------------------------------------------
    # Build graph
    # --------------------------------------------------

    G = nx.from_pandas_edgelist(
        top_edges,
        source=source,
        target=target,
        edge_attr=weight,
        create_using=nx.Graph(),
    )

    # --------------------------------------------------
    # Identify hubs
    # --------------------------------------------------

    if hub_metric == "degree":

        hub_scores = dict(G.degree())

    elif hub_metric == "weighted_degree":

        hub_scores = dict(
            G.degree(weight=weight)
        )

    else:

        raise ValueError(
            "hub_metric must be "
            "'degree' or 'weighted_degree'"
        )

    hubs = sorted(
        hub_scores,
        key=hub_scores.get,
        reverse=True,
    )[:n_hubs]

    # --------------------------------------------------
    # Helper: determine whether two line segments cross
    # --------------------------------------------------

    def segments_cross(p1, p2, p3, p4):

        def orientation(a, b, c):
            return np.cross(
                b - a,
                c - a
            )

        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)

        return (
            o1 * o2 < 0
            and
            o3 * o4 < 0
        )

    # --------------------------------------------------
    # Count edge crossings
    # --------------------------------------------------

    def count_crossings(pos):

        edges = list(G.edges())
        crossings = 0

        for i in range(len(edges)):

            u1, v1 = edges[i]

            for j in range(i + 1, len(edges)):

                u2, v2 = edges[j]

                # Edges sharing a node are not crossings
                if len({
                    u1, v1,
                    u2, v2
                }) < 4:
                    continue

                if segments_cross(
                    np.array(pos[u1]),
                    np.array(pos[v1]),
                    np.array(pos[u2]),
                    np.array(pos[v2]),
                ):
                    crossings += 1

        return crossings

    # --------------------------------------------------
    # Search many spring layouts
    # --------------------------------------------------

    best_pos = None
    best_score = np.inf
    best_crossings = None

    for trial in range(n_layout_trials):

        rng = np.random.default_rng(
            seed + trial
        )

        # Random starting positions
        pos0 = {
            n: rng.uniform(-1, 1, size=2)
            for n in G.nodes
        }

        # Start hubs close to origin
        for hub in hubs:
            pos0[hub] = rng.normal(
                0,
                0.03,
                size=2,
            )

        pos_trial = nx.spring_layout(
            G,
            pos=pos0,
            seed=seed + trial,
            weight=weight,
            k=spring_k,
            iterations=iterations,
        )

        # --------------------------------------------------
        # Recenter whole layout around origin
        # --------------------------------------------------

        xy = np.array(
            list(pos_trial.values())
        )

        center = xy.mean(axis=0)

        pos_trial = {
            node: xy - center
            for node, xy
            in pos_trial.items()
        }

        # --------------------------------------------------
        # Count crossings
        # --------------------------------------------------

        crossings = count_crossings(
            pos_trial
        )

        # --------------------------------------------------
        # Penalize hubs far from center
        # --------------------------------------------------

        hub_distance = np.mean([
            np.linalg.norm(
                pos_trial[h]
            )
            for h in hubs
        ])

        score = (
            crossings
            +
            hub_center_strength
            * hub_distance
        )

        if score < best_score:

            best_score = score
            best_pos = pos_trial
            best_crossings = crossings

    pos = best_pos

    # --------------------------------------------------
    # Edge widths
    # --------------------------------------------------

    weights = np.array([
        G[u][v][weight]
        for u, v in G.edges()
    ])

    if len(weights) > 0:

        if weights.max() > weights.min():

            widths = (
                min_edge_width
                +
                (
                    max_edge_width
                    - min_edge_width
                )
                *
                (
                    weights - weights.min()
                )
                /
                (
                    weights.max()
                    - weights.min()
                )
            )

        else:

            widths = np.full(
                len(weights),
                (
                    min_edge_width
                    + max_edge_width
                ) / 2
            )

    else:

        widths = []

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize
    )

    # invisible nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=0,
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        alpha=edge_alpha,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=font_size,
        ax=ax,
    )

    ax.axis("off")
    ax.margins(0.05)

    plt.tight_layout()

    print(
        f"{G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )

    print(
        "Hubs:",
        hubs
    )

    print(
        "Edge crossings:",
        best_crossings
    )

    return G, pos, hubs, top_edges