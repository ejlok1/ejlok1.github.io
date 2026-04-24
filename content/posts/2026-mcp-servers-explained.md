---
title: "MCP Servers: The USB Standard That AI Was Missing"
date: 2026-04-07
description: "Model Context Protocol is the emerging standard for connecting LLMs to tools and data. Here's what it is, how it works, and why it matters — with a real production example."
tags: ["agentic-ai", "MCP", "LLM", "tools", "NVIDIA", "cuOpt", "production"]
categories: ["New Series"]
showToc: true
---

## The Integration Problem

If you've worked with LLMs in production, you've hit the integration wall.

You have a model that can reason beautifully. You have tools — APIs, databases, services — that hold the data and capabilities the model needs. The question is: **how do they talk to each other?**

For a while, everyone rolled their own answer. LangChain. Custom function-calling schemas. Bespoke glue code that worked for one model and broke when you switched to another.

This is exactly where USB was before USB: every device had its own connector. Printers, keyboards, cameras — each manufacturer did it differently. Connecting a new device required knowing exactly which port it needed.

**MCP (Model Context Protocol)** is the USB standard that AI was missing.

## What MCP Actually Is

MCP is an open standard — developed by Anthropic and now broadly adopted — that defines a common way for LLMs to discover and interact with external tools and data sources.

The architecture is simple:

```
LLM / Agent
     │
     ▼
 MCP Client  ←──── your application
     │
     ▼
 MCP Server  ←──── your tools, APIs, data
```

An **MCP Server** exposes a set of **tools** — named functions with defined input schemas and return formats. The LLM can discover available tools, decide which to call, pass in parameters, and receive structured results.

The key properties:
- **Standardised**: Any MCP-compatible LLM can use any MCP-compatible server
- **Discoverable**: Tools declare themselves — the LLM doesn't need to be pre-programmed to know what's available
- **Composable**: Multiple MCP servers can be connected simultaneously — file system + database + external API, all available to one agent

## A Real Example: cuOpt + MCP

I'll give you a concrete production example. At DataRobot, I built an MCP server wrapping **NVIDIA cuOpt** — a GPU-accelerated solver for routing and optimisation problems.

The use case: a logistics company wants to optimise delivery routes across a city. Historically this required an operations research team to formulate the problem and run it manually. With the MCP server:

1. A user describes the problem in natural language: *"I have 12 trucks, 156 deliveries, and need to minimise total distance while respecting 9am–5pm delivery windows."*
2. The LLM (Claude) recognises this as a Vehicle Routing Problem
3. It calls the `cuopt_solve` tool via MCP, passing the fleet and order data
4. cuOpt solves the problem (GPU-accelerated, seconds not hours)
5. The LLM receives the solution and presents it as a human-readable plan with map routes

The MCP server definition for the tool looks like this:

```python
@server.tool()
async def cuopt_solve(
    fleet_data: dict,
    order_data: dict,
    solver_config: dict = None
) -> dict:
    """
    Solve a Vehicle Routing Problem using NVIDIA cuOpt.
    
    Args:
        fleet_data: Vehicle definitions (count, capacity, start/end locations)
        order_data: Delivery orders (locations, time windows, demand)
        solver_config: Optional solver parameters (time limit, objective)
    
    Returns:
        Optimized routes per vehicle with total cost metrics
    """
    # ... cuOpt API call ...
    return solution
```

The LLM doesn't need to know anything about cuOpt's internals. It just sees: *"there is a tool called cuopt_solve that takes fleet and order data and returns routes."* It figures out the rest from context.

## Why This Is Different from Function Calling

You might say: *"We already had function calling / tool use in OpenAI and Anthropic APIs. What's new?"*

Fair question. The difference is **scope and standardisation**:

| | Function Calling | MCP |
|---|---|---|
| Where tools are defined | In your application code | In a standalone server |
| Reusability | Tied to one app | Any MCP-compatible client can use it |
| Discovery | You pass a list | Server exposes tool manifest |
| Transport | In-process | Over network (HTTP, stdio) |
| Ecosystem | Fragmented | Growing standard |

MCP servers can run locally (stdio transport) or over a network. They can be shared across teams. They can be versioned and deployed independently.

## The Ecosystem in 2026

The MCP ecosystem has grown fast. There are now MCP servers for:

- **Databases**: PostgreSQL, SQLite, Snowflake, BigQuery
- **File systems**: local FS, Google Drive, SharePoint
- **APIs**: GitHub, Slack, Jira, Salesforce, Google Calendar
- **Specialised solvers**: cuOpt (routing), Gurobi (MILP)
- **Developer tools**: browser control, code execution, shell access

Claude Code ships with built-in MCP support. The major agent frameworks (LangGraph, AutoGen, CrewAI) all support MCP. The standard has won.

## What to Build First

If you're an engineer starting to work with agents, my recommendation:

1. **Start with a local stdio server** — wrap something you already use (a script, an API you know)
2. **Expose 3–5 well-defined tools** with clear schemas and docstrings
3. **Test with Claude Desktop** — it has built-in MCP client support and you can iterate quickly
4. **Then add complexity** — multiple servers, tool chaining, error handling

The hardest part isn't the technology. It's **tool design**. Tools need clear names, clear inputs, and clear outputs. Ambiguous tool definitions confuse LLMs just like they confuse humans.

Next post: I'll show the full cuOpt MCP server codebase — building it from scratch, deploying it, and solving the classic Travelling Salesman Problem across all 47 Japanese prefectures.

---

*Eu Jin Lok — Melbourne, April 2026*
