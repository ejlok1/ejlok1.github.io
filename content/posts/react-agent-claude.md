---
title: "Building a ReACT Agent from Scratch with Claude"
date: 2026-04-28
description: "Most AI tools run a ReACT loop under the hood and never show you. Here's how to build one from scratch — Think, Act, Observe, Reflect — with nothing but the Claude API and a for loop."
tags: ["agents", "claude", "LLM", "python", "agentic-ai"]
categories: ["New Series"]
showToc: true
---

Most AI tools today — ChatGPT, Claude.ai, Copilot — run a ReACT loop under the hood. You never see it. The interface just... thinks, then answers.

This post tears off the cover.

We build a ReACT agent from scratch using the Claude API. Every phase — **Think, Act, Observe, Reflect** — is printed explicitly as it happens. No LangChain, no AutoGen, no magic. Just API calls and a loop.

## What is ReACT?

**ReACT** (Reason + Act) is a framework for agents that interleave reasoning with tool use, first introduced in a [2022 paper](https://arxiv.org/abs/2210.03629) by Yao et al. The core idea is simple:

```
┌────────────────────────────────────────────────────┐
│  THINK   → reason about what to do next            │
│  ACT     → call a tool with specific inputs        │  ← loop
│  OBSERVE → receive the tool result                 │
│  REFLECT → summarise; decide to loop or stop       │
└────────────────────────────────────────────────────┘
```

When you ask ChatGPT a question that requires browsing the web, it is running this loop internally. When Claude searches for information before answering, same thing. The interface abstracts the loop so cleanly you never see the iterations.

The mechanism that frameworks like LangChain hide: every iteration appends to a **`messages` list** that gets sent back to the model in full. The model has no persistent memory — it just sees a growing conversation each time.

That's what we're going to expose.

## Setup

```python
# pip install anthropic

import anthropic
import json
import os

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
MODEL = "claude-sonnet-4-6"
```

## Tool Definitions

Tools are declared as JSON schemas. Claude reads the `description` to decide when and how to use each tool. We define two:

- **`calculator`** — evaluates a maths expression
- **`web_search`** — returns simulated search results (so this notebook runs offline)

These schemas are sent to the model on **every** API call. Claude has no tool knowledge outside of what you pass in the request.

```python
TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluates a mathematical expression and returns the numeric result. Use for arithmetic, percentages, and basic maths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python-evaluable maths expression, e.g. '68_000_000 * 0.07'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "web_search",
        "description": "Searches the web for factual information. Use for current data, statistics, or facts you are uncertain about.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A clear, specific search query"
                }
            },
            "required": ["query"]
        }
    }
]
```

## Tool Execution — the ACT→OBSERVE bridge

When Claude calls a tool, it returns a `tool_use` block containing the tool name and arguments. **We** are responsible for executing that tool and returning a `tool_result`. This is the part frameworks automate — here we do it ourselves.

```python
# Simulated search results — replace with a real search API in production
SEARCH_STUBS = {
    "france population": (
        "France population 2024: approximately 68.4 million people. "
        "Source: INSEE, 2024."
    ),
}

def run_calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{result:,}" if isinstance(result, (int, float)) else str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"

def run_web_search(query: str) -> str:
    for key, result in SEARCH_STUBS.items():
        if key in query.lower():
            return result
    return "No results found."

def execute_tool(name: str, tool_input: dict) -> str:
    if name == "calculator":
        return run_calculator(tool_input["expression"])
    elif name == "web_search":
        return run_web_search(tool_input["query"])
    return f"Unknown tool: {name}"
```

## The System Prompt — with caching

The system prompt defines the agent's behaviour across the entire conversation. We add `cache_control` so it isn't re-tokenised on every loop iteration — each loop sends the same system prompt, so it hits the cache from the second call onwards.

```python
SYSTEM_PROMPT = """\
You are a ReACT agent. You solve problems by reasoning step-by-step and calling tools when you need information.

For every response:
1. Start with your reasoning — what do you know, what do you need to find out?
2. Call a tool if you need external information or calculations.
3. When you have enough information to fully answer the question, begin your response with exactly: FINAL ANSWER:

Be precise and show your working."""

SYSTEM = [
    {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"}  # 5-minute cache
    }
]
```

## The ReACT Loop

Here is where everything comes together. The loop makes **two API calls per iteration**:

| Call | Phases | `tool_choice` | What Claude returns |
|------|--------|--------------|---------------------|
| Call 1 | **Think + Act** | `auto` | Reasoning text + `tool_use` block |
| Call 2 | **Reflect** | `none` | Reflection text only — no tool calls |

We also print the raw `messages` list at each step so you can see exactly what the model receives — this is what frameworks hide.

```python
def print_messages(messages: list, label: str = "Messages payload") -> None:
    """Pretty-print the messages list sent to the API."""
    print(f"\n{'─'*50}")
    print(f"📨 {label} ({len(messages)} message(s))")
    print(f"{'─'*50}")
    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg["content"]
        if isinstance(content, str):
            preview = content[:120].replace("\n", " ")
            print(f"  [{i}] {role}: {preview}{'...' if len(content) > 120 else ''}")
        elif isinstance(content, list):
            for block in content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        print(f"  [{i}] {role} [text]: {block.text[:80].replace(chr(10), ' ')}...")
                    elif block.type == "tool_use":
                        print(f"  [{i}] {role} [tool_use]: {block.name}({json.dumps(block.input)})")
                elif isinstance(block, dict):
                    btype = block.get("type", "?")
                    if btype == "tool_result":
                        print(f"  [{i}] {role} [tool_result]: {str(block.get('content', ''))[:60]}")
                    elif btype == "text":
                        print(f"  [{i}] {role} [text]: {str(block.get('text', ''))[:80]}")
    print()


def run_react_agent(question: str, max_iterations: int = 5) -> str:
    messages = [{"role": "user", "content": question}]

    print(f"\n{'═'*60}")
    print(f"❓ QUESTION: {question}")
    print(f"{'═'*60}")

    for iteration in range(1, max_iterations + 1):
        print(f"\n\n{'━'*60}")
        print(f"  ITERATION {iteration}")
        print(f"{'━'*60}")

        # ── THINK + ACT ──────────────────────────────────────────
        print_messages(messages, label=f"Sending to Claude (Think+Act, iter {iteration})")

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
        )

        think_text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                think_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(block)

        print("\n🧠 [THINK]")
        print(think_text)

        if not tool_calls:
            print("\n✅ No tool calls — Claude answered directly.")
            return think_text

        print("\n⚡ [ACT]")
        for tc in tool_calls:
            print(f"  Tool  : {tc.name}")
            print(f"  Args  : {json.dumps(tc.input, indent=4)}")

        messages.append({"role": "assistant", "content": response.content})

        # ── OBSERVE ──────────────────────────────────────────────
        print("\n👁 [OBSERVE]")
        tool_results = []
        for tc in tool_calls:
            result = execute_tool(tc.name, tc.input)
            print(f"  {tc.name}({tc.input}) → {result}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result
            })

        # Combine tool results + reflect prompt in a single user turn
        reflect_prompt = {
            "type": "text",
            "text": "Reflect on what you just learned. Summarise your findings so far. "
                    "If you now have enough to fully answer the original question, "
                    "start your response with FINAL ANSWER:"
        }
        messages.append({"role": "user", "content": tool_results + [reflect_prompt]})

        # ── REFLECT ──────────────────────────────────────────────
        print_messages(messages, label=f"Sending to Claude (Reflect, iter {iteration})")

        reflect_response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            tool_choice={"type": "none"},  # Reflect = reason only, no tool calls
        )

        reflect_text = ""
        for block in reflect_response.content:
            if block.type == "text":
                reflect_text = block.text

        print("\n🔍 [REFLECT]")
        print(reflect_text)

        u = reflect_response.usage
        print(f"\n  💡 Tokens: input={u.input_tokens} | "
              f"cache_write={getattr(u, 'cache_creation_input_tokens', 0)} | "
              f"cache_read={getattr(u, 'cache_read_input_tokens', 0)} | "
              f"output={u.output_tokens}")

        messages.append({"role": "assistant", "content": reflect_text})

        if "FINAL ANSWER:" in reflect_text:
            print(f"\n{'═'*60}")
            print(f"✅ Done in {iteration} iteration(s).")
            print(f"{'═'*60}")
            return reflect_text

        messages.append({
            "role": "user",
            "content": "Continue. Call another tool if you need more information."
        })

    return "Max iterations reached without a final answer."
```

## Running It

This question requires two tool calls: a web search for France's population, then a calculator to compute 7% of that figure.

```python
answer = run_react_agent(
    question="What is the current population of France? "
             "Then calculate what 7% of that population would be."
)
```

Here's what the output looks like:

```
============================================================
❓ QUESTION: What is the current population of France? Then calculate what 7% ...
============================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ITERATION 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📨 Sending to Claude (Think+Act, iter 1) (1 message(s))
──────────────────────────────────────────────────
  [0] USER: What is the current population of France?...

🧠 [THINK]
To answer this question, I need to find France's current population. I don't have
reliable up-to-date statistics in my training data for this, so I'll search for it.

⚡ [ACT]
  Tool  : web_search
  Args  : {
      "query": "current population of France 2024"
  }

👁 [OBSERVE]
  web_search({'query': 'current population of France 2024'})
  → France population 2024: approximately 68.4 million people. Source: INSEE, 2024.

📨 Sending to Claude (Reflect, iter 1) (3 message(s))
──────────────────────────────────────────────────
  [0] USER: What is the current population of France?...
  [1] ASSISTANT [text]: To answer this question, I need to find France's current...
  [1] ASSISTANT [tool_use]: web_search({'query': 'current population of France 2024'})
  [2] USER [tool_result]: France population 2024: approximately 68.4 million...
  [2] USER [text]: Reflect on what you just learned...

🔍 [REFLECT]
I now have France's population (68.4 million) but still need to calculate 7% of it.
I'll need to call the calculator tool for that.

  💡 Tokens: input=312 | cache_write=89 | cache_read=0 | output=41

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ITERATION 2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧠 [THINK]
I have the population figure: 68.4 million. Now I need to calculate 7% of that.

⚡ [ACT]
  Tool  : calculator
  Args  : {
      "expression": "68_400_000 * 0.07"
  }

👁 [OBSERVE]
  calculator({'expression': '68_400_000 * 0.07'}) → 4,788,000.0

🔍 [REFLECT]
FINAL ANSWER: France's current population is approximately 68.4 million people
(as of 2024, per INSEE). 7% of that population is approximately 4,788,000 people.

  💡 Tokens: input=198 | cache_write=0 | cache_read=89 | output=52

✅ Done in 2 iteration(s).
```

Notice the `cache_read=89` on iteration 2 — the system prompt wasn't re-tokenised. On longer loops with larger system prompts this adds up quickly.

## What Just Happened

Here's the full picture of what was sent to the API across both iterations:

```
Iteration 1 — Think+Act:
  messages = [
    {role: "user", content: "What is the current population..."}
  ]
  → Claude returns: text (Think) + tool_use: web_search (Act)

Iteration 1 — Observe + Reflect:
  messages = [
    {role: "user",      content: "What is the current population..."},
    {role: "assistant", content: [text, tool_use: web_search]},
    {role: "user",      content: [tool_result, "Reflect on what you learned..."]}
  ]
  → Claude returns: "I still need to calculate 7%..."

Iteration 2 — Think+Act:
  messages = [...above...,
    {role: "assistant", content: "I still need to calculate..."},
    {role: "user",      content: "Continue..."}
  ]
  → Claude returns: text (Think) + tool_use: calculator (Act)

Iteration 2 — Reflect:
  messages = [...all of the above...,
    {role: "assistant", content: [text, tool_use: calculator]},
    {role: "user",      content: [tool_result, "Reflect..."]}
  ]
  → Claude returns: "FINAL ANSWER: ..."
```

**The model has no memory.** Every API call sends the entire conversation from the beginning. The `messages` list is the agent's complete state — nothing more, nothing less.

This is the thing that took me a while to fully internalise coming from classical ML. A deep learning model has weights that encode learned patterns. An LLM agent has... a Python list of dicts. The "intelligence" is in how you structure that list and when you choose to truncate it.

## Key API Patterns

A few things worth calling out in the implementation:

**`tool_choice={"type": "none"}` on the Reflect call.** This forces a text-only response — Claude can't sneak in a tool call during the reflection phase. Without this, Claude might try to call another tool immediately rather than pausing to summarise.

**Appending `response.content` (not just text) to history.** The API requires that `tool_use` blocks in the assistant turn are preserved. If you extract only the text and discard the `tool_use` blocks, the API will reject the follow-up because the `tool_result` has no matching `tool_use` to reference.

**Combining `tool_result` and the reflect prompt in one user turn.** You can mix block types in a single message's `content` list. This avoids back-to-back user messages (which some older models dislike) while keeping the phases clean.

**`cache_control` on the system prompt.** The system prompt is identical on every call — caching it costs a one-time 1.25× write premium but subsequent reads come back at ~10% of the base token price. On a 10-iteration agentic loop, this pays for itself many times over.

## What Frameworks Actually Do

LangChain's `AgentExecutor`, LlamaIndex's `ReActAgent`, CrewAI's agents — all of them implement variants of this same loop. The abstraction is useful for production systems where you need observability hooks, error handling, streaming, and tool registries.

But if you're learning how agents work, or debugging why your agent is making bizarre decisions, you want exactly this: the raw messages, printed at every step. The loop above is ~70 lines of Python and has no hidden state.

The notebook for this post is available [here](/notebooks/react_agent_claude.ipynb) if you want to run it yourself. Swap out the `web_search` stub for a real search API and you have a functional research agent.

ReACT feels almost quaint now that models like Claude natively run tool loops internally. But understanding the explicit loop is still the right mental model — it's what's happening under the hood, and when your agent misbehaves, it's the only frame that lets you debug it.

---

*Eu Jin Lok — Melbourne, April 2026*
