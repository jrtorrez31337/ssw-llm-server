# NPC Agent Integration Guide

## Platform Overview

The SSW AI inference platform is accessible at the gateway host on port `8000` via OpenAI-compatible API.

| Tier | Model | Best For |
|------|-------|----------|
| `heavy` | Qwen2.5-14B-Instruct-AWQ | Complex reasoning, lore-heavy NPCs, quest-critical dialogue |
| `light` | Qwen3-8B-AWQ | High-frequency, low-latency interactions, ambient chatter |

**Max context window: 64,000 tokens** (input + output combined)
**API:** `POST /v1/chat/completions` — fully OpenAI SDK compatible

---

## Known Issues — Action Required

Confirmed across multiple agents in audits 2026-02-17 and 2026-02-18. These are implementation bugs in `npc-agent-ts`, not model issues. Both must be fixed.

---

### BUG 1: `course.engage` is never called

`course.set_destination` has been observed repeating 5–22 consecutive ticks across every agent tested. Agents set a destination but never engage it, so the ship never moves.

**Root cause:** The agent loop lets the LLM decide every tick, and the LLM re-decides `set_destination` rather than completing the sequence. `course.engage` must be enforced by the loop, not left to the model.

**Required fix — enforce engage in the tick loop:**

```typescript
let pendingEngage = false;

async function tick(agentState: AgentState): Promise<Action> {
  // Intercept: if last tick was set_destination, this tick must be engage
  if (pendingEngage) {
    pendingEngage = false;
    return { intent: 'course.engage' };
  }

  const action = await llm.decide(agentState);

  if (action.intent === 'course.set_destination') {
    pendingEngage = true;  // next tick will force engage
  }

  return action;
}
```

Do not rely on the model to call `course.engage` unprompted — it will not. Enforce it in code.

---

### BUG 2: No stuck detection

When an agent becomes stuck (same intent 3+ consecutive ticks), it continues calling the LLM indefinitely with the same context and getting the same answer. This wastes inference capacity and degrades latency for all other agents.

**Required fix — track intent history and break loops:**

```typescript
const STUCK_THRESHOLD = 3;
const recentIntents: string[] = [];

async function tick(agentState: AgentState): Promise<Action> {
  let action = await llm.decide(agentState);

  // Update intent history
  recentIntents.push(action.intent);
  if (recentIntents.length > STUCK_THRESHOLD) recentIntents.shift();

  // Detect stuck: same intent repeated STUCK_THRESHOLD times
  const isStuck =
    recentIntents.length === STUCK_THRESHOLD &&
    recentIntents.every(i => i === recentIntents[0]);

  if (isStuck) {
    recentIntents.length = 0;
    // Re-decide with explicit reassessment instruction injected into context
    action = await llm.decide({
      ...agentState,
      systemOverride: 'You have been repeating the same action. Stop. Reassess your situation and choose a different action.',
    });
  }

  return action;
}
```

Reset `recentIntents` on any successful state change (e.g., sector transition, new event received).

---

## Token Budget

64K is shared between everything in the request. Design for the common case — most NPC interactions should target **under 8K total** to preserve throughput.

| Zone | Tokens | Contents |
|------|--------|----------|
| System prompt | 1,000–3,000 | Persona, world rules, behavioral constraints |
| World context | 0–8,000 | Zone state, active quest flags, nearby entities |
| Conversation history | 0–48,000 | Prior turns (or summary — see below) |
| Player input | 0–2,000 | Current message |
| Output reserve | 512–4,000 | NPC response |

Reserve the large budget for NPCs that carry long quest arcs or persistent relationships. Don't burn it on ambient interactions.

---

## System Prompt Structure

Use explicit labeled sections. Bullet lists over paragraphs. Dense prose wastes tokens and degrades instruction following.

```
IDENTITY
You are [NPC_NAME], [role in one sentence]. [Personality in one sentence.]

WORLD STATE
Location: [zone name]
Faction: [affiliation]
Active flags: [quest_id: status, ...]

BEHAVIORAL RULES
- [constraint 1]
- [constraint 2]

KNOWLEDGE LIMITS
You know: [what this NPC should know]
You do not know: [explicit gaps — prevents hallucination]
```

**Do:**
- Put hard constraints before personality
- Be explicit about knowledge gaps
- Reference faction/quest state by ID, not prose

**Don't:**
- Include full lore documents — summarize or chunk
- Restate world context that's already in message history
- Use vague descriptors ("friendly," "wise") without behavioral anchors

---

## Model Selection

### Use `heavy` when:
- NPC carries quest-critical information where accuracy matters
- Multi-step reasoning is required (evaluating player actions, generating consequences)
- Long-form narrative generation (journal entries, lore reveals)
- Infrequent, high-value interactions

### Use `light` when:
- Ambient, flavor, or one-line responses
- High concurrency (merchants, guards, crowd NPCs)
- Player is spam-pinging the same NPC
- Latency matters more than depth

### Request format:
```json
{
  "model": "heavy",
  "messages": [...],
  "max_tokens": 512,
  "temperature": 0.7
}
```

---

## Conversation History Management

Don't accumulate raw history indefinitely. KV cache pressure slows inference for all concurrent users.

Observed: `test-agent-01` accumulated 760 memory entries with 0 compaction events. Memory growing unbounded will bloat prompts and increase latency for all agents.

**Compaction triggers — fire on EITHER condition:**

```typescript
const MEMORY_COUNT_THRESHOLD = 100;   // entries
const MEMORY_TOKEN_THRESHOLD = 6_000; // estimated prompt tokens

function shouldCompact(memoryEntries: MemoryEntry[], estimatedTokens: number): boolean {
  return memoryEntries.length >= MEMORY_COUNT_THRESHOLD ||
         estimatedTokens >= MEMORY_TOKEN_THRESHOLD;
}
```

**Summarize-and-compress:**

```typescript
if (shouldCompact(memory, estimatedTokens)) {
  const summary = await llm.complete({
    model: 'light',
    messages: [
      {
        role: 'system',
        content: 'Summarize this NPC memory into key facts, entity relationships, and unresolved threads. Max 300 tokens. Be dense.',
      },
      { role: 'user', content: serializeMemory(memory) },
    ],
    max_tokens: 300,
  });

  // Replace full memory with single compressed entry
  memory = [{ role: 'assistant', content: `[Compressed memory]: ${summary}` }];
}
```

For persistent NPCs, store structured state (relationship score, known facts, quest flags) in your world DB — not in the conversation history.

---

## Tool Calling

Both models support hermes-format tool calling. The gateway passes tool definitions through unchanged.

```json
{
  "model": "heavy",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "query_world_state",
        "description": "Query current state for a zone or entity",
        "parameters": {
          "type": "object",
          "properties": {
            "zone_id": {"type": "string", "description": "Zone identifier"},
            "entity_type": {
              "type": "string",
              "enum": ["player", "faction", "quest", "npc"]
            }
          },
          "required": ["zone_id"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

**Note:** The `light` model has thinking mode disabled at the gateway level. Do not attempt to enable it — behavior is undefined and will likely degrade output quality.

---

## Client Example (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<gateway-host>:8000/v1",
    api_key="not-used"  # internal network, no auth
)

response = client.chat.completions.create(
    model="light",
    messages=[
        {
            "role": "system",
            "content": (
                "IDENTITY\n"
                "You are Mira, a dockworker in Port Veluna. You trade gossip for coin.\n\n"
                "BEHAVIORAL RULES\n"
                "- Stay in character\n"
                "- Do not break immersion\n"
                "- You have never heard of the Inner Sanctum\n\n"
                "KNOWLEDGE LIMITS\n"
                "You know: shipping schedules, dock crew names, recent arrivals\n"
                "You do not know: anything about the player's quest, inner city politics"
            )
        },
        {
            "role": "user",
            "content": "Have you seen any strange ships lately?"
        }
    ],
    max_tokens=256,
    temperature=0.8
)

print(response.choices[0].message.content)
```

Streaming:
```python
stream = client.chat.completions.create(
    model="heavy",
    messages=[...],
    max_tokens=1024,
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Quick Reference

| Parameter | Value |
|-----------|-------|
| Max context (input + output) | 64,000 tokens |
| Safe max input | ~60,000 tokens |
| Recommended output reserve | 512–2,048 tokens |
| Gateway endpoint | `:8000/v1/chat/completions` |
| Available models | `heavy`, `light` |
| Streaming | Supported (`"stream": true`) |
| Authentication | None (internal network) |
| Tool calling | Supported (hermes format, both models) |

---

## Performance & Latency

### Expected Latency (awq_marlin kernel active)

| Model | Prompt Size | Expected P50 |
|-------|-------------|--------------|
| `light` (8B) | ~4K tokens | 8–12s |
| `light` (8B) | ~8K tokens | 12–18s |
| `heavy` (14B) | ~4K tokens | 15–20s |
| `heavy` (14B) | ~8K tokens | 20–30s |

If you observe P50 > 30s, the Marlin kernel may not be active. Check worker startup logs for:
```
Detected that the model can run with awq_marlin
```
If present, ensure `--quantization=awq_marlin` is set (not `--quantization=awq`).

### Route by Decision Type

**Use `light` for:**
- All navigation decisions (`course.*`, `navigate.*`, `scan.*`, `warp.*`)
- Ambient actions with no player interaction
- High-frequency ticks

**Use `heavy` for:**
- Player-facing dialogue where quality matters
- Quest-critical reasoning or lore reveals
- Tool calls that require multi-step logic

Routing navigation to `heavy` when `light` suffices doubles your latency for no gain.

---

## Action Sequences & Lifecycles

Actions are not self-completing — many require a follow-up action in the next tick to take effect. Failing to complete a sequence causes stuck behavior.

### Navigation (travel to a destination)
```
course.set_destination  →  course.engage  →  [in transit]  →  next decision
```
**Never** repeat `course.set_destination` without first calling `course.engage`. Setting a destination does not start movement.

### Course correction mid-travel
```
course.cancel  →  course.set_destination  →  course.engage
```

### Combat engagement
```
combat.set_target  →  combat.power_distribution  →  combat.fire_weapon
```
Target acquisition alone does not fire. Power must be allocated first.

### Station operations
```
navigate.dock  →  station.refuel / station.repair  →  navigate.undock
```

### Warp travel
```
warp.set_factor  →  course.set_destination  →  course.engage
```
Set warp factor before destination for hyperspace jumps.

---

## Behavioral Anti-Patterns

Observed in audits 2026-02-17 and 2026-02-18. Confirmed across multiple agents.

### Stuck destination loop
**Symptom:** `course.set_destination` fires 5+ consecutive ticks with no movement.
**Cause:** Agent sets a destination but never calls `course.engage`.
**Fix:** See **BUG 1** and **BUG 2** at the top of this document — both fixes are required.

### Boilerplate chat responses
**Symptom:** Identical chat replies sent to multiple different players or situations.
**Cause:** Agent treats chat as a template fill rather than a contextual response.
**Fix:** Include the player's name, their message content, and the current situation in the response. Vary acknowledgments. Never cache or reuse a prior reply verbatim.

### Over-routing to `heavy`
**Symptom:** Long LLM latency on routine navigation ticks; `heavy` usage for non-dialogue decisions.
**Fix:** Route by decision type — see Performance section above. Navigation agents should use `light` by default.

### Ignoring available actions
**Symptom:** Agent never uses `combat.fire_weapon`, `navigate.jump`, `warp.set_factor`, etc. despite game state warranting them.
**Cause:** System prompt doesn't motivate these behaviors, or action definitions lack enough context.
**Fix:** Ensure system prompt describes *when* to use each action class. Include explicit behavioral triggers (e.g., "if under attack and health < 50%, consider retreat via `navigate.jump`").

---

## Rollout Checklist

Before deploying a new NPC agent:

- [ ] System prompt tested under 3K tokens
- [ ] Model tier selected (`heavy` vs `light`) with justification — navigation agents default to `light`
- [ ] History management strategy defined (window size or summarization trigger)
- [ ] Tool definitions validated against API contract
- [ ] Max output tokens set explicitly (don't rely on model defaults)
- [ ] Behavior tested at edge: unknown questions, off-topic inputs, repeated queries
- [ ] Action sequences validated: multi-step actions (destination→engage, dock→repair→undock) tested end-to-end
- [ ] Stuck detection accounted for: agent has a rule to break out of repeated identical intents
- [ ] P50 latency measured under expected load and within target for chosen model tier
