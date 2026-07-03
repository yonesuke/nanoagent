# -*- coding: utf-8 -*-
"""Interactive HTML visualization for agent configuration and execution traces.

Generates a self-contained HTML file with zero external dependencies.
Open in any browser to explore agent trees and execution timelines.

Features:
- Left panel: collapsible agent configuration tree (tools, sub-agents)
- Right panel: execution trace timeline with timestamps and elapsed times
- Linked panels: click a node to filter events, click an event to highlight its node
- Event type legend with visibility toggles
- Per-node duration stats
- Dark mode (respects system preference)

Usage::

    from nanoagent.contrib.viz import render_html

    render_html(agent, result, output_path="trace.html")
    # Open trace.html in your browser.

If only ``agent`` is provided (no ``result``), shows the configuration tree only.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from nanoagent.agent import Agent, RunResult
from nanoagent.events import Event, EventType
from nanoagent.node import Node
from nanoagent.tools import Tool

# ── Helpers ──────────────────────────────────────────────────────────────


def _tool_to_dict(tool: Tool) -> dict[str, Any]:
    """Serialize a Tool to a JSON-safe dict."""
    return {
        "name": tool.name,
        "description": tool.description,
    }


def _agent_to_dict(agent: Agent) -> dict[str, Any]:
    """Walk the agent tree and serialize to a JSON-safe dict.

    Sub-agents (Agent instances used as tools) are discovered by
    introspecting tools that return an Agent from as_tool()/handoff().
    Since tools are opaque callables, we approximate: any Agent in the
    tree that has this agent as parent is a sub-agent.
    """
    # Collect all agents in the tree starting from this root
    agents: list[Agent] = []

    def walk(node: Agent) -> None:
        agents.append(node)

    walk(agent)

    # Build tree
    def build(agt: Agent) -> dict[str, Any]:
        children = [a for a in agents if a.parent is agt and a is not agt]
        return {
            "id": agt.id,
            "name": agt.name,
            "depth": agt.depth,
            "instructions": agt.instructions[:200] if agt.instructions else "",
            "tools": [_tool_to_dict(t) for t in agt.tools],
            "children": [build(c) for c in children],
        }

    return build(agent)


def _event_to_dict(ev: Event) -> dict[str, Any]:
    """Serialize a single Event to a JSON-safe dict."""
    d: dict[str, Any] = {
        "type": ev.type.value,
        "node_id": ev.node_id,
        "node_path": ev.node_path,
        "depth": ev.depth,
        "name": ev.name,
    }
    if ev.timestamp is not None:
        d["timestamp"] = ev.timestamp
        dt = datetime.fromtimestamp(ev.timestamp, tz=timezone.utc)
        d["time_iso"] = dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
    if ev.elapsed_ms is not None:
        d["elapsed_ms"] = round(ev.elapsed_ms, 1)
    if ev.delta:
        d["delta"] = ev.delta
    if ev.tool_call_id:
        d["tool_call_id"] = ev.tool_call_id
    if ev.tool_args:
        d["tool_args"] = ev.tool_args
    if ev.tool_result:
        preview = ev.tool_result if len(ev.tool_result) <= 500 else ev.tool_result[:500] + "..."
        d["tool_result"] = preview
    if ev.question:
        d["question"] = ev.question
    if ev.error:
        d["error"] = ev.error
    if ev.meta:
        d["meta"] = ev.meta
    return d


def _compute_node_stats(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute per-node statistics from events.

    Returns:
        {node_id: {name, depth, event_count, start_time, end_time, duration_ms}}
    """
    stats: dict[str, dict[str, Any]] = {}
    # Track NODE_START / NODE_END per node_id
    node_starts: dict[str, dict[str, Any]] = {}

    for ev in events:
        nid = ev["node_id"]
        if nid not in stats:
            stats[nid] = {
                "name": ev.get("name", ""),
                "depth": ev.get("depth", 0),
                "event_count": 0,
                "start_time": None,
                "end_time": None,
                "duration_ms": None,
            }
        stats[nid]["event_count"] += 1

        if ev["type"] == "node_start":
            node_starts[nid] = ev
            if stats[nid]["start_time"] is None:
                ts = ev.get("timestamp")
                if ts:
                    stats[nid]["start_time"] = ts
        elif ev["type"] == "node_end":
            ts = ev.get("timestamp")
            if ts:
                stats[nid]["end_time"] = ts
                if stats[nid]["start_time"] is not None:
                    stats[nid]["duration_ms"] = round(
                        (ts - stats[nid]["start_time"]) * 1000, 1
                    )

    return stats


def _format_duration(ms: float) -> str:
    """Format milliseconds into a human-readable string."""
    if ms < 1:
        return f"{ms * 1000:.0f}μs"
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60_000:
        return f"{ms / 1000:.1f}s"
    return f"{ms / 60_000:.1f}m"


def _format_tool_args(args: dict[str, Any]) -> str:
    """Format tool args for inline display."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 60:
            s = s[:60] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


# ── HTML template ────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
:root {{
    --bg: #fff;
    --bg-secondary: #f8f9fa;
    --bg-hover: #e9ecef;
    --bg-active: #d0ebff;
    --text: #212529;
    --text-secondary: #6c757d;
    --border: #dee2e6;
    --accent: #228be6;
    --accent-bg: #e7f5ff;
    --node-start: #40c057;
    --node-end: #868e96;
    --text-delta: #212529;
    --tool-call: #fd7e14;
    --tool-result: #15aabf;
    --human-input: #be4bdb;
    --error: #e03131;
    --error-bg: #fff5f5;
}}
@media (prefers-color-scheme: dark) {{
    :root {{
        --bg: #1a1b1e;
        --bg-secondary: #25262b;
        --bg-hover: #2c2e33;
        --bg-active: #1c3a5c;
        --text: #ced4da;
        --text-secondary: #868e96;
        --border: #373a40;
        --accent: #4dabf7;
        --accent-bg: #1c3a5c;
        --error-bg: #2d1b1b;
    }}
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}}
header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-secondary);
    font-size: 13px;
    flex-shrink: 0;
}}
header h1 {{ font-size: 14px; font-weight: 600; }}
header .summary {{ color: var(--text-secondary); font-size: 12px; }}
.main {{
    display: flex;
    flex: 1;
    overflow: hidden;
}}
.panel {{
    overflow-y: auto;
    padding: 12px;
}}
.left {{
    width: 320px;
    flex-shrink: 0;
    border-right: 1px solid var(--border);
    background: var(--bg-secondary);
}}
.right {{
    flex: 1;
}}
.tree, .trace {{ font-size: 13px; line-height: 1.6; }}
.tree-node {{ cursor: pointer; user-select: none; }}
.tree-node:hover > .tree-label {{ background: var(--bg-hover); border-radius: 4px; }}
.tree-label {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 4px;
    border-radius: 4px;
    transition: background 0.1s;
}}
.tree-label.active {{ background: var(--bg-active); }}
.tree-children {{ margin-left: 16px; border-left: 1px dashed var(--border); padding-left: 8px; }}
.tree-children.collapsed {{ display: none; }}
.tree-toggle {{
    display: inline-block;
    width: 12px;
    font-size: 10px;
    color: var(--text-secondary);
    cursor: pointer;
    text-align: center;
    flex-shrink: 0;
}}
.tree-toggle.empty {{ visibility: hidden; }}
.tree-icon {{ font-size: 12px; flex-shrink: 0; }}
.tree-name {{ font-weight: 500; }}
.tree-tools {{ color: var(--text-secondary); font-size: 11px; margin-left: 4px; }}
.tree-detail {{
    font-size: 11px;
    color: var(--text-secondary);
    margin-left: 20px;
    padding: 2px 0;
}}
.trace-event {{
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 2px 4px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.1s;
}}
.trace-event:hover {{ background: var(--bg-hover); }}
.trace-event.highlight {{ background: var(--bg-active); }}
.trace-time {{ color: var(--text-secondary); font-size: 11px; width: 48px; flex-shrink: 0; text-align: right; }}
.trace-elapsed {{ color: var(--text-secondary); font-size: 10px; width: 42px; flex-shrink: 0; text-align: right; }}
.trace-indent {{ flex-shrink: 0; color: var(--border); font-size: 11px; }}
.trace-icon {{ font-size: 12px; flex-shrink: 0; width: 16px; text-align: center; }}
.trace-content {{ flex: 1; min-width: 0; word-break: break-word; }}
.trace-delta {{ color: inherit; }}
.trace-text-block {{
    border-left: 2px solid var(--border);
    margin: 2px 0;
    padding-left: 6px;
}}
.trace-text-block .trace-content {{
    color: var(--text-secondary);
    font-style: italic;
    white-space: pre-wrap;
}}
.trace-tool {{ color: var(--tool-call); }}
.trace-result {{ color: var(--tool-result); }}
.trace-error {{ color: var(--error); }}
.legend {{
    display: flex;
    gap: 12px;
    padding: 8px 16px;
    border-top: 1px solid var(--border);
    font-size: 11px;
    flex-shrink: 0;
    flex-wrap: wrap;
    background: var(--bg-secondary);
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 4px;
    cursor: pointer;
    opacity: 0.6;
    transition: opacity 0.15s;
    user-select: none;
}}
.legend-item.on {{ opacity: 1; }}
.legend-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}}
.legend-dot.node_start {{ background: var(--node-start); }}
.legend-dot.node_end {{ background: var(--node-end); }}
.legend-dot.text_delta {{ background: var(--text-delta); border: 1px solid var(--border); }}
.legend-dot.tool_call {{ background: var(--tool-call); }}
.legend-dot.tool_result {{ background: var(--tool-result); }}
.legend-dot.human_input_request {{ background: var(--human-input); }}
.legend-dot.error {{ background: var(--error); }}
.empty-state {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-secondary);
    font-size: 14px;
    gap: 8px;
}}
h3 {{ font-size: 13px; font-weight: 600; margin-bottom: 8px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }}
</style>
</head>
<body>
<header>
    <h1>{title}</h1>
    <div class="summary" id="header-summary"></div>
</header>
<div class="main">
    <div class="panel left" id="config-panel">
        <h3>Agent Tree</h3>
        <div class="tree" id="config-tree"></div>
    </div>
    <div class="panel right" id="trace-panel">
        <h3>Execution Trace</h3>
        <div class="trace" id="trace-list"></div>
    </div>
</div>
<div class="legend" id="legend"></div>
<script>
const DATA = {data_json};

const EVENT_ICONS = {{
    node_start: "▶",
    node_end: "◀",
    text_delta: "",
    tool_call: "🔧",
    tool_result: "📥",
    human_input_request: "❓",
    error: "❌",
}};

const EVENT_COLORS = {{
    node_start: "var(--node-start)",
    node_end: "var(--node-end)",
    text_delta: "var(--text)",
    tool_call: "var(--tool-call)",
    tool_result: "var(--tool-result)",
    human_input_request: "var(--human-input)",
    error: "var(--error)",
}};

let activeNodeId = null;
let visibleEventTypes = new Set(Object.keys(EVENT_ICONS));

function $fmtDuration(ms) {{
    if (ms == null) return "";
    if (ms < 1) return Math.round(ms * 1000) + "μs";
    if (ms < 1000) return Math.round(ms) + "ms";
    if (ms < 60000) return (ms / 1000).toFixed(1) + "s";
    return (ms / 60000).toFixed(1) + "m";
}}

function $fmtTime(iso) {{
    return iso || "";
}}

function $fmtToolArgs(args) {{
    if (!args) return "";
    return Object.entries(args).map(([k, v]) => {{
        const s = String(v);
        return k + "=" + (s.length > 60 ? s.slice(0, 60) + "..." : s);
    }}).join(", ");
}}

// ── Config tree ─────────────────────────────────────────────────────────

function buildConfigTree(node, container) {{
    const hasChildren = node.children && node.children.length > 0;
    const hasTools = node.tools && node.tools.length > 0;
    const hasExpand = hasChildren || hasTools;

    const div = document.createElement("div");
    div.className = "tree-node";
    div.dataset.nodeId = node.id;

    const label = document.createElement("span");
    label.className = "tree-label";

    const toggle = document.createElement("span");
    toggle.className = "tree-toggle" + (hasExpand ? "" : " empty");
    toggle.textContent = hasExpand ? "▼" : " ";
    label.appendChild(toggle);

    const icon = document.createElement("span");
    icon.className = "tree-icon";
    icon.textContent = "🤖";
    label.appendChild(icon);

    const name = document.createElement("span");
    name.className = "tree-name";
    name.textContent = node.name + " ";
    label.appendChild(name);

    const depth = document.createElement("span");
    depth.className = "tree-detail";
    depth.textContent = "(depth=" + node.depth + ")";
    label.appendChild(depth);

    // Duration
    if (DATA.node_stats && DATA.node_stats[node.id] && DATA.node_stats[node.id].duration_ms != null) {{
        const dur = document.createElement("span");
        dur.className = "tree-detail";
        dur.textContent = " ⏱ " + $fmtDuration(DATA.node_stats[node.id].duration_ms);
        label.appendChild(dur);
    }}

    div.appendChild(label);

    // Children container
    const childrenDiv = document.createElement("div");
    childrenDiv.className = "tree-children";

    // Node children
    if (hasChildren) {{
        for (const child of node.children) {{
            buildConfigTree(child, childrenDiv);
        }}
    }}

    // Tools
    if (hasTools) {{
        for (const tool of node.tools) {{
            const tdiv = document.createElement("div");
            tdiv.className = "tree-detail";
            tdiv.style.marginLeft = "20px";
            tdiv.textContent = "🔧 " + tool.name;
            if (tool.description) {{
                const desc = tool.description.length > 60
                    ? tool.description.slice(0, 60) + "..."
                    : tool.description;
                tdiv.textContent += " — " + desc;
            }}
            childrenDiv.appendChild(tdiv);
        }}
    }}

    div.appendChild(childrenDiv);

    // Click handlers
    label.addEventListener("click", (e) => {{
        e.stopPropagation();
        if (hasExpand) {{
            childrenDiv.classList.toggle("collapsed");
            toggle.textContent = childrenDiv.classList.contains("collapsed") ? "▶" : "▼";
        }}
        // Filter trace by this node
        activateNode(node.id, label);
    }});

    // Instructions tooltip
    if (node.instructions) {{
        label.title = node.instructions;
    }}

    container.appendChild(div);
}}

function activateNode(nodeId, labelEl) {{
    // Deactivate previous
    if (activeNodeId) {{
        document.querySelectorAll(".tree-label.active").forEach(el => el.classList.remove("active"));
    }}
    activeNodeId = nodeId;
    if (labelEl) labelEl.classList.add("active");
    filterTrace();
}}

// ── Trace ────────────────────────────────────────────────────────────────

function filterTrace() {{
    const events = document.querySelectorAll(".trace-event");
    for (const el of events) {{
        const evNodeId = el.dataset.nodeId;
        const evType = el.dataset.eventType;

        const typeOk = visibleEventTypes.has(evType);
        const nodeOk = !activeNodeId || evNodeId === activeNodeId;
        el.style.display = (typeOk && nodeOk) ? "" : "none";
    }}
}}

function buildTrace(container) {{
    if (!DATA.events || DATA.events.length === 0) {{
        container.innerHTML = '<div class="empty-state"><span>No events</span></div>';
        return;
    }}

    // Helper: flush a text delta buffer as a single row
    function flushDeltaBuf(buf) {{
        if (buf.length === 0) return null;
        const first = buf[0];
        const combined = buf.map(e => e.delta || "").join("");
        const row = document.createElement("div");
        row.className = "trace-event trace-text-block";
        row.dataset.nodeId = first.node_id;
        row.dataset.eventType = "text_delta";

        const timeEl = document.createElement("span");
        timeEl.className = "trace-time";
        timeEl.textContent = first.time_iso ? first.time_iso.slice(0, 8) : "";
        row.appendChild(timeEl);

        const elapsedEl = document.createElement("span");
        elapsedEl.className = "trace-elapsed";
        elapsedEl.textContent = first.elapsed_ms != null && first.elapsed_ms > 0
            ? "+" + $fmtDuration(first.elapsed_ms)
            : "";
        row.appendChild(elapsedEl);

        const indentEl = document.createElement("span");
        indentEl.className = "trace-indent";
        indentEl.textContent = "│ ".repeat(Math.max(0, first.depth || 0));
        row.appendChild(indentEl);

        const iconEl = document.createElement("span");
        iconEl.className = "trace-icon";
        iconEl.textContent = "";
        row.appendChild(iconEl);

        const contentEl = document.createElement("span");
        contentEl.className = "trace-content trace-delta";
        contentEl.textContent = combined.length > 500 ? combined.slice(0, 500) + "..." : combined;
        row.appendChild(contentEl);

        row.addEventListener("click", () => activateNode(first.node_id));
        return row;
    }}

    // Build a single trace-event row for non-text events
    function makeRow(ev) {{
        const row = document.createElement("div");
        row.className = "trace-event";
        row.dataset.nodeId = ev.node_id;
        row.dataset.eventType = ev.type;

        const timeEl = document.createElement("span");
        timeEl.className = "trace-time";
        timeEl.textContent = ev.time_iso ? ev.time_iso.slice(0, 8) : "";
        row.appendChild(timeEl);

        const elapsedEl = document.createElement("span");
        elapsedEl.className = "trace-elapsed";
        elapsedEl.textContent = ev.elapsed_ms != null && ev.elapsed_ms > 0
            ? "+" + $fmtDuration(ev.elapsed_ms)
            : "";
        row.appendChild(elapsedEl);

        const indentEl = document.createElement("span");
        indentEl.className = "trace-indent";
        indentEl.textContent = "│ ".repeat(Math.max(0, ev.depth || 0));
        row.appendChild(indentEl);

        const iconEl = document.createElement("span");
        iconEl.className = "trace-icon";
        iconEl.textContent = EVENT_ICONS[ev.type] || "";
        row.appendChild(iconEl);

        const contentEl = document.createElement("span");
        contentEl.className = "trace-content";

        switch (ev.type) {{
            case "node_start":
                contentEl.textContent = "[" + (ev.name || "?") + "] started";
                contentEl.style.color = EVENT_COLORS.node_start;
                break;
            case "node_end":
                contentEl.textContent = "[" + (ev.name || "?") + "] finished";
                contentEl.style.color = EVENT_COLORS.node_end;
                break;
            case "tool_call":
                contentEl.textContent = (ev.name || "tool") + "(" + ($fmtToolArgs(ev.tool_args) || "") + ")";
                contentEl.className += " trace-tool";
                break;
            case "tool_result":
                contentEl.textContent = (ev.tool_result || "").slice(0, 200);
                contentEl.className += " trace-result";
                break;
            case "human_input_request":
                contentEl.textContent = "asks: " + (ev.question || "").slice(0, 150);
                contentEl.style.color = EVENT_COLORS.human_input_request;
                break;
            case "error":
                contentEl.textContent = (ev.error || "").slice(0, 200);
                contentEl.className += " trace-error";
                break;
        }}
        row.appendChild(contentEl);

        row.addEventListener("click", () => activateNode(ev.node_id));
        return row;
    }}

    // Main loop: coalesce consecutive text_delta events
    let deltaBuf = [];
    for (const ev of DATA.events) {{
        if (ev.type === "text_delta") {{
            deltaBuf.push(ev);
        }} else {{
            // Flush buffered deltas
            const deltaRow = flushDeltaBuf(deltaBuf);
            if (deltaRow) container.appendChild(deltaRow);
            deltaBuf = [];
            // Emit non-delta event
            container.appendChild(makeRow(ev));
        }}
    }}
    // Flush remaining
    const deltaRow = flushDeltaBuf(deltaBuf);
    if (deltaRow) container.appendChild(deltaRow);
}}

// ── Legend ───────────────────────────────────────────────────────────────

function buildLegend(container) {{
    const types = [
        ["node_start", "Start"],
        ["node_end", "End"],
        ["text_delta", "Text"],
        ["tool_call", "Tool Call"],
        ["tool_result", "Result"],
        ["human_input_request", "Human"],
        ["error", "Error"],
    ];
    for (const [type, label] of types) {{
        const item = document.createElement("span");
        item.className = "legend-item on";
        item.dataset.legendType = type;

        const dot = document.createElement("span");
        dot.className = "legend-dot " + type;
        item.appendChild(dot);

        const text = document.createElement("span");
        text.textContent = label;
        item.appendChild(text);

        item.addEventListener("click", () => {{
            if (visibleEventTypes.has(type)) {{
                visibleEventTypes.delete(type);
                item.classList.remove("on");
            }} else {{
                visibleEventTypes.add(type);
                item.classList.add("on");
            }}
            filterTrace();
        }});

        container.appendChild(item);
    }}
}}

// ── Header summary ───────────────────────────────────────────────────────

function buildSummary() {{
    const parts = [];
    if (DATA.events) {{
        parts.push(DATA.events.length + " events");
    }}
    if (DATA.node_stats) {{
        const nNodes = Object.keys(DATA.node_stats).length;
        parts.push(nNodes + " nodes");
    }}
    document.getElementById("header-summary").textContent = parts.join(" · ");
}}

// ── Init ─────────────────────────────────────────────────────────────────

buildConfigTree(DATA.config, document.getElementById("config-tree"));
buildTrace(document.getElementById("trace-list"));
buildLegend(document.getElementById("legend"));
buildSummary();
</script>
</body>
</html>"""


# ── Public API ───────────────────────────────────────────────────────────


def render_html(
    agent: Agent,
    result: RunResult | None = None,
    *,
    output_path: str = "trace.html",
    title: str | None = None,
) -> str:
    """Generate a self-contained interactive HTML visualization.

    Args:
        agent: The root agent (configuration tree is built from this).
        result: Optional :class:`RunResult` from a completed agent run.
            If provided, the execution trace is shown alongside the config tree.
        output_path: Where to write the HTML file.
        title: Page title (defaults to agent name + " · nanoagent").

    Returns:
        The absolute path to the generated HTML file.
    """
    if title is None:
        title = f"{agent.name} · nanoagent"

    config = _agent_to_dict(agent)

    data: dict[str, Any] = {"config": config}

    if result is not None:
        events_list = [_event_to_dict(ev) for ev in result.events]
        data["events"] = events_list
        data["node_stats"] = _compute_node_stats(events_list)
    else:
        data["events"] = None
        data["node_stats"] = None

    data_json = json.dumps(data, ensure_ascii=False)

    html = _HTML_TEMPLATE.format(title=title, data_json=data_json)

    out = pathlib.Path(output_path)
    out.write_text(html, encoding="utf-8")

    return str(out.resolve())