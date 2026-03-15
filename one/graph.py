"""Live knowledge graph visualization.

Serves a self-contained HTML page with a d3.js force-directed graph
of entities and their co-occurrence relationships. Fetches data from
the local SQLite store.
"""

from . import store


def get_graph_data() -> dict:
    """Build the nodes/edges dict from the entity store.

    Returns:
        {
            "nodes": [{"id": int, "name": str, "type": str, "count": int}, ...],
            "edges": [{"source": int, "target": int, "weight": int}, ...]
        }
    """
    entities = store.get_entities(limit=200)
    nodes = []
    name_to_id = {}

    for ent in entities:
        node_id = ent["id"]
        nodes.append({
            "id": node_id,
            "name": ent["name"],
            "type": ent["type"],
            "count": ent["observation_count"],
        })
        name_to_id[ent["name"]] = node_id

    edges = []
    seen_edges = set()

    for ent in entities:
        related = store.get_related_entities(ent["name"], limit=20)
        source_id = name_to_id.get(ent["name"])
        if source_id is None:
            continue
        for rel in related:
            target_id = name_to_id.get(rel["name"])
            if target_id is None:
                continue
            edge_key = tuple(sorted((source_id, target_id)))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append({
                "source": source_id,
                "target": target_id,
                "weight": rel["shared_memories"],
            })

    return {"nodes": nodes, "edges": edges}


def get_entity_memories(entity_name: str) -> list[dict]:
    """Return memories linked to an entity, formatted for the API."""
    memories = store.get_memories_for_entity(entity_name, limit=30)
    return [
        {
            "id": m["id"],
            "text": m["raw_text"],
            "source": m["source"],
            "timestamp": m["timestamp"],
            "label": m["tm_label"],
        }
        for m in memories
    ]


GRAPH_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>one — knowledge graph</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #0a0a0f;
    color: #c0c0c0;
    font-family: 'Courier New', monospace;
    overflow: hidden;
    width: 100vw;
    height: 100vh;
}
svg { width: 100vw; height: 100vh; display: block; }
.node-label {
    fill: #999;
    font-size: 10px;
    pointer-events: none;
    user-select: none;
}
#panel {
    position: fixed;
    top: 16px;
    right: 16px;
    width: 320px;
    max-height: calc(100vh - 32px);
    background: rgba(15, 15, 25, 0.92);
    border: 1px solid #333;
    border-radius: 6px;
    padding: 16px;
    overflow-y: auto;
    display: none;
    font-size: 12px;
    line-height: 1.5;
    z-index: 10;
}
#panel h2 {
    color: #00e5ff;
    font-size: 14px;
    margin-bottom: 8px;
    word-break: break-all;
}
#panel .meta { color: #666; margin-bottom: 12px; }
#panel .memory {
    border-top: 1px solid #222;
    padding: 8px 0;
}
#panel .memory .src { color: #888; font-size: 10px; }
#panel .memory .txt { color: #aaa; font-size: 11px; margin-top: 2px; }
#panel .close-btn {
    position: absolute;
    top: 8px;
    right: 12px;
    cursor: pointer;
    color: #666;
    font-size: 16px;
}
#panel .close-btn:hover { color: #fff; }
#hud {
    position: fixed;
    bottom: 16px;
    left: 16px;
    color: #444;
    font-size: 11px;
    z-index: 10;
}
</style>
</head>
<body>
<svg id="graph"></svg>
<div id="panel">
    <span class="close-btn" onclick="closePanel()">&times;</span>
    <h2 id="panel-name"></h2>
    <div class="meta" id="panel-meta"></div>
    <div id="panel-memories"></div>
</div>
<div id="hud">one knowledge graph</div>

<script>
const TYPE_COLORS = {
    concept: '#00e5ff',
    file: '#ff00ff',
    tool: '#ffff00',
    rule: '#00ff88',
    pattern: '#ff8800',
};
const DEFAULT_COLOR = '#666';

const svg = d3.select('#graph');
const width = window.innerWidth;
const height = window.innerHeight;

svg.attr('viewBox', [0, 0, width, height]);

const g = svg.append('g');

// Zoom
svg.call(d3.zoom()
    .scaleExtent([0.1, 8])
    .on('zoom', (event) => g.attr('transform', event.transform))
);

const simulation = d3.forceSimulation()
    .force('link', d3.forceLink().id(d => d.id).distance(80))
    .force('charge', d3.forceManyBody().strength(-120))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 4));

let linkGroup = g.append('g').attr('class', 'links');
let nodeGroup = g.append('g').attr('class', 'nodes');
let labelGroup = g.append('g').attr('class', 'labels');

function nodeRadius(d) {
    return Math.max(4, Math.min(20, 3 + Math.sqrt(d.count || 1) * 2));
}

function nodeColor(d) {
    return TYPE_COLORS[d.type] || DEFAULT_COLOR;
}

function update(data) {
    const nodes = data.nodes || [];
    const edges = data.edges || [];

    // Build id set for filtering edges
    const idSet = new Set(nodes.map(n => n.id));
    const validEdges = edges.filter(e => idSet.has(e.source?.id ?? e.source) && idSet.has(e.target?.id ?? e.target));

    // Links
    const link = linkGroup.selectAll('line').data(validEdges, d => `${d.source?.id ?? d.source}-${d.target?.id ?? d.target}`);
    link.exit().remove();
    const linkEnter = link.enter().append('line')
        .attr('stroke', '#222')
        .attr('stroke-width', d => Math.max(0.5, Math.min(3, d.weight * 0.5)));
    const allLinks = linkEnter.merge(link);

    // Nodes
    const node = nodeGroup.selectAll('circle').data(nodes, d => d.id);
    node.exit().remove();
    const nodeEnter = node.enter().append('circle')
        .attr('r', nodeRadius)
        .attr('fill', nodeColor)
        .attr('stroke', d => nodeColor(d))
        .attr('stroke-width', 0.5)
        .attr('fill-opacity', 0.7)
        .attr('cursor', 'pointer')
        .on('click', (event, d) => showEntityPanel(d))
        .call(d3.drag()
            .on('start', dragStart)
            .on('drag', dragging)
            .on('end', dragEnd));
    const allNodes = nodeEnter.merge(node);
    allNodes.attr('r', nodeRadius).attr('fill', nodeColor);

    // Labels
    const label = labelGroup.selectAll('text').data(nodes, d => d.id);
    label.exit().remove();
    const labelEnter = label.enter().append('text')
        .attr('class', 'node-label')
        .attr('dx', d => nodeRadius(d) + 4)
        .attr('dy', '0.35em')
        .text(d => d.name.length > 24 ? d.name.slice(0, 22) + '..' : d.name);
    const allLabels = labelEnter.merge(label);

    simulation.nodes(nodes);
    simulation.force('link').links(validEdges);
    simulation.alpha(0.3).restart();

    simulation.on('tick', () => {
        allLinks
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        allNodes
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
        allLabels
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });

    document.getElementById('hud').textContent =
        `one knowledge graph | ${nodes.length} entities | ${validEdges.length} connections`;
}

function dragStart(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
}
function dragging(event, d) { d.fx = event.x; d.fy = event.y; }
function dragEnd(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null; d.fy = null;
}

function showEntityPanel(d) {
    const panel = document.getElementById('panel');
    document.getElementById('panel-name').textContent = d.name;
    document.getElementById('panel-meta').textContent = `type: ${d.type} | observations: ${d.count}`;
    document.getElementById('panel-memories').innerHTML = 'loading...';
    panel.style.display = 'block';

    fetch(`/api/entity/${encodeURIComponent(d.name)}/memories`)
        .then(r => r.json())
        .then(memories => {
            const container = document.getElementById('panel-memories');
            if (!memories.length) {
                container.innerHTML = '<div style="color:#555">no linked memories</div>';
                return;
            }
            container.innerHTML = memories.map(m =>
                `<div class="memory">
                    <div class="src">${m.source} | ${m.label} | ${m.timestamp.slice(0, 19)}</div>
                    <div class="txt">${escapeHtml(m.text.slice(0, 300))}</div>
                </div>`
            ).join('');
        })
        .catch(() => {
            document.getElementById('panel-memories').innerHTML = '<div style="color:#555">fetch error</div>';
        });
}

function closePanel() {
    document.getElementById('panel').style.display = 'none';
}

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function fetchGraph() {
    fetch('/api/graph')
        .then(r => r.json())
        .then(data => update(data))
        .catch(err => console.error('graph fetch error:', err));
}

// Initial load and periodic refresh
fetchGraph();
setInterval(fetchGraph, 10000);
</script>
</body>
</html>
"""
