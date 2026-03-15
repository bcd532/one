# one

Persistent memory for AI coding tools.

## What it does

- **Remembers across sessions** -- conversations, decisions, file paths, and technical context are stored locally in SQLite and recalled automatically when relevant.
- **Learns your rules** -- repeated preferences ("no floating point in core", "always run tests") are detected, organized into a tree, and injected into future conversations.
- **Gates intelligently** -- an Active Inference-inspired gate scores each message for novelty and information value before storing, so noise stays out.
- **Runs locally** -- no cloud, no API keys, no external services required. Works out of the box with just the Claude CLI.

## Architecture

```
                       +------------------+
                       |    one (TUI)     |
                       |  Textual app.py  |
                       +--------+---------+
                                |
             +------------------+------------------+
             |                  |                  |
      +------+------+   +------+------+   +-------+------+
      | ClaudeProxy  |   | AIF Gate   |   | Rule Tree    |
      | stdin/stdout |   | excitation |   | activation   |
      | stream JSON  |   | + novelty  |   | + learning   |
      +------+------+   +------+------+   +-------+------+
             |                  |                  |
             v                  v                  v
      +------+------------------+------------------+------+
      |              Storage Backend (backend.py)          |
      +---+--------------------------------------------+---+
          |                                            |
   +------+------+                          +----------+--------+
   | SqliteBackend|                          | FoundryBackend    |
   | ~/.one/one.db|                          | (optional)        |
   +------+------+                          +----------+--------+
          |                                            |
   +------+------+                          +----------+--------+
   | HDC Encoder  |                          | Palantir AIP      |
   | 4096-dim     |                          | Ontology API      |
   | trigram+word  |                          +-------------------+
   +--------------+
```

## Requirements

- Python 3.10+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-cli) installed and authenticated

## Install

```bash
git clone https://github.com/bcd532/one.git
cd one
pip install -e .
```

## Usage

```bash
one
```

This launches a terminal UI wrapping Claude with persistent memory. Your conversations, decisions, and context are stored locally and recalled automatically.

### Commands

| Command       | Description                                |
|---------------|--------------------------------------------|
| `/rules`      | Show the active rule tree for this project |
| `/rule <text>`| Manually add a rule                        |
| `/recall`     | Force a memory recall for recent context   |
| `/cost`       | Show cumulative session cost and duration   |
| `/clear`      | Clear the chat display                     |
| `/quit`       | Exit                                       |

### CLI flags

```
one "prompt"              # send an initial prompt
one -m sonnet             # use a different model
one -c                    # continue last session
one --no-foundry          # force local-only mode
one -d /path/to/project   # set working directory
```

## How memory works

1. **Encode** -- every message is encoded into a 4096-dimensional hypervector using character trigrams, word vectors, and word bigrams (HDC algebra).
2. **Gate** -- an Active Inference gate scores the message for novelty, content quality, and information type. Low-value messages (acknowledgments, routine tool output) are dropped.
3. **Store** -- qualifying messages are stored in SQLite with their HDC vector, source metadata, and a conversation regime tag.
4. **Recall** -- on topic shifts or periodic intervals, the most relevant stored memories are retrieved via cosine similarity and injected as context.
5. **Learn** -- high-confidence decisions and repeated preferences are promoted to rules, organized in a tree that activates based on current context.

## Optional: Foundry backend

For cross-machine persistent memory via Palantir AIP:

```bash
pip install -e ".[foundry]"
```

Configure credentials:

```bash
mkdir -p ~/.one
echo "host=your-foundry-hostname" > ~/.one/config
echo "your-token" > ~/.one/token
```

Or via environment variables:

```bash
export ONE_FOUNDRY_HOST=your-foundry-hostname
export FOUNDRY_TOKEN=your-token
```

## Optional: Local condensation with Gemma

When [Ollama](https://ollama.com) is installed with `gemma3:4b`, retrieved memories are condensed into compact context blocks before injection, reducing token usage.

```bash
ollama pull gemma3:4b
```

No configuration needed -- `one` detects it automatically and falls back to raw injection when unavailable.

## License

MIT
