# one

A knowledge graph orchestrator for Claude Code. Persistent memory, autonomous research, cross-domain reasoning, and verification -- building toward a world where AI coding tools don't forget, don't regress, and don't hallucinate.

> **Alpha.** Core memory works. Autonomous features are experimental. Active development.

## The Vision

Every Claude session starts from zero. It doesn't remember what you built yesterday. It doesn't know what broke last week. It doesn't learn your patterns. And when it writes code, it hallucinates function names, wrong column names, broken signatures -- because it has no grounded truth about your codebase.

`one` is building the home Claude lives in. A knowledge graph it traverses to find answers. A memory it recalls to avoid past mistakes. A verification engine that catches errors before they land. A research system that discovers connections across everything you've ever worked on.

The dream: no hallucinations. No forgetting. Research, proof, validation, shipped. An LLM that stops regressing to its training distribution and starts building on verified, project-specific truth.

## What works today

**Persistent memory** -- conversations, decisions, and context stored in SQLite with 4096-dimensional Hyperdimensional Computing (HDC) vector embeddings. Recalled automatically by cosine similarity when the topic shifts or periodically during conversation.

**Knowledge graph** -- entities (files, concepts, tools, code patterns, people, orgs) extracted from every conversation and linked to memories. The graph grows with every session.

**Rule learning** -- repeated preferences detected across sessions, organized into a contextual activation tree, injected per-turn based on what you're currently working on. The system learns how you want to work.

**Autonomous research** -- `/morgoth` runs an indefinite loop: research a topic with Claude, challenge every finding through dialectic (thesis/antithesis/synthesis), mine for contradictions, synthesize cross-domain patterns, build code from verified findings, verify with the engine, prune weak claims, iterate. Each cycle strengthens the knowledge graph.

**Zero Hallucination Engine** -- AST-parses every code edit, extracts SQL queries, checks column and table names against live database schemas. Maps every function, call, and file dependency in the codebase. Detects when edits break callers in other files. Blocks bad code before it saves.

**Foundry sync** -- push the entire knowledge graph to Palantir AIP as ontology objects for enterprise-scale visualization and querying.

## What happened when I let it run

I pointed `/morgoth` at a trading system codebase. Over 6 iterations (~16 hours), it autonomously:

- Produced **420 research findings** with cited academic sources
- Mined **472 contradictions**, resolved 14 with structured reasoning
- Deprecated **589 weak findings** through adversarial challenge
- Discovered **21 universal structural patterns** across domains
- Generated **27 cross-domain syntheses** connecting software engineering, molecular biology, quantitative finance, and AI security

The system connected Python's lazy import pattern to RNA transcription -- both are deferred materialization architectures where dormant capabilities are suppressed until activation context arrives. It formalized why certain bug classes are invisible to quality checks (manifold membership). It coined "trust laundering" for when unverified data passes through enough processing layers to appear verified.

These connections emerged from the system arguing with itself, not from directed prompts. The dialectic engine challenges every finding, the contradiction miner finds conflicts, and weak claims get pruned. What survives multiple iterations of adversarial challenge is harder knowledge.

## Roadmap

**Active Inference (AIF)** -- the current gating function is inspired by AIF but isn't a real implementation. I'm working toward actual free energy minimization with a generative model, so the system can compute genuine surprise relative to its beliefs and update those beliefs from evidence. The field names in the codebase (`aif_confidence`, `regime_tag`) are the scaffolding for this.

**Tsetlin Machines (TM)** -- the `tm_label` field is currently a string tag. The goal is real Tsetlin Machine classification for memory categorization -- interpretable Boolean logic that can explain WHY a memory was categorized a certain way, not just that it was.

**Epistemic safety** -- LLM confidence scoring is circular (the LLM scores its own output). Working on provenance tracking, confidence ceilings, and circular reference detection so the knowledge graph can distinguish between empirically grounded claims and LLM speculation.

**Foundry link integration** -- objects push to Foundry, but entity-to-memory links need action type configuration. Working on making the full ontology graph navigable in Vertex.

## Install

```bash
git clone https://github.com/bcd532/one.git
cd one
pip install -e .
```

Requires Python 3.10+ and [Claude CLI](https://docs.anthropic.com/en/docs/claude-cli).

## Usage

```bash
one                        # launch TUI
one "prompt"               # with initial prompt
one -m sonnet              # different model
one -c                     # continue last session
one -d /path/to/project    # set working directory
```

### Commands

| Command | Description |
|---------|-------------|
| `/recall` | Force memory recall |
| `/rules` | Show rule tree |
| `/stats` | Memory store statistics |
| `/entities` | Show knowledge graph |
| `/search <q>` | Search memories |
| `/auto <goal>` | Autonomous agent -- Claude drives until done |
| `/morgoth <goal>` | Research + build + verify loop |
| `/research <topic>` | Deep research with gap analysis |
| `/synthesize` | Cross-domain insight generation |
| `/verify` | Run Zero Hallucination Engine |
| `/health` | Knowledge graph health metrics |
| `/audit` | Quality audit |

### CLI

```bash
one map       # map codebase: functions, calls, deps
one verify    # verify all files against live schemas
one ground    # populate verified ground truths
one sync      # push knowledge graph to Foundry
one server    # REST API on :4111
```

## Architecture

```
one (TUI) --> ClaudeProxy --> claude CLI subprocess
    |
    +--> Knowledge Engines
    |      research, dialectic, synthesis
    |      contradictions, analogy, verification
    |      experiments, morgoth, swarm
    |
    +--> Zero Hallucination Engine
    |      AST parsing, SQL verification
    |      codebase ontology, impact analysis
    |
    +--> Storage
           SQLite (local, always works)
           Palantir Foundry (optional, enterprise)
           HDC encoder (4096-dim vectors)
```

## How memory works

1. **Encode** -- HDC algebra: character trigrams + word vectors + bigrams into 4096-dim hypervector
2. **Gate** -- score for novelty, content quality, information type. Drop noise.
3. **Store** -- SQLite with vector, metadata, regime tag
4. **Recall** -- cosine similarity retrieval on topic shift or periodic interval
5. **Learn** -- repeated preferences become rules in a contextual activation tree

## Pictures
![le tui](./picture.png)

## Contributing

This is early. The codebase is 37 Python files and ~17,000 lines. If you're interested in persistent memory for AI tools, knowledge graph reasoning, HDC encoding, or autonomous research systems -- contributions welcome. Start with the issues or the roadmap above.

## License

MIT
