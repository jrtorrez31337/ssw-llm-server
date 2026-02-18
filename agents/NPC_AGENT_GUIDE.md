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

**Why this happens:** The LLM decision is stateless. Each tick it receives a snapshot of the world and picks the most appropriate action from that snapshot alone — it has no memory of what it decided last tick. When the state shows "destination: [X], status: not yet engaged," the model evaluates this as a task to be done and emits `set_destination` again, because from its perspective that is the correct response to an unengaged destination. It cannot infer "I must have set this last tick, therefore I should engage now" — that reasoning requires memory of prior decisions, which is not in its context. Documenting the sequence in the guide or the system prompt does not fix this, because the model has no way to observe that it's been looping. The sequence must be enforced in application state.

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

**Why this happens:** A stuck agent is sending identical requests — same system prompt, same world state snapshot — and receiving identical responses. The model isn't confused; it's doing exactly what it should given the context. Nothing in the context signals that time has passed or that the same action has already been tried. The fix is not to improve the prompt — it's to detect the loop in application code and inject new information that the model hasn't seen before, breaking the repetition.

**Why threshold of 3:** At current P50 latency (~50s), threshold 3 catches a loop within ~2.5 minutes. Lower would produce false positives on legitimately repeated actions (e.g., `scan.active` while surveying a sector). Higher allows more wasted inference calls before intervention.

**Why this degrades other agents:** At P50 ~50s, a single stuck agent fires roughly 1.2 requests/minute. The gateway has 2 workers per model tier. One stuck agent on `light` represents ~60% of one worker's capacity — directly queuing other agents behind it. The more stuck agents, the worse the tail latency for everyone.

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
    // Re-decide with explicit reassessment instruction injected into context.
    // The override works because it adds information the model hasn't seen —
    // "you've been repeating yourself" — which is sufficient to break the loop.
    action = await llm.decide({
      ...agentState,
      systemOverride: 'You have been repeating the same action. Stop. Reassess your situation and choose a different action.',
    });
  }

  return action;
}
```

Reset `recentIntents` on any successful state change (e.g., sector transition, new event received).

**This applies to all intents, not just `course.set_destination`.** Observed in audit 2026-02-18: `scan.active` stuck in 26 separate runs of 5+ consecutive ticks. The sliding window must wrap the entire decision loop, not be applied selectively per intent. A dedicated stuck handler per intent will miss cases — use a single generic check on the returned intent string.

---

### BUG 3: Model returning non-JSON / empty intent

Observed in audit 2026-02-18: 2x `Failed to parse LLM response: no valid JSON found`, 2x empty intent string `""` blocked by the safety gate.

**Why this happens:** Two likely triggers:
- **Memory compaction summaries injecting unusual context.** The compaction output is fed back as an assistant message. If the summary contains structured text that looks like a partial response, the model may continue it rather than starting a fresh JSON action.
- **System override messages.** The stuck-detection override (`"You have been repeating the same action..."`) is a user-role injection mid-loop. Some models respond to this with acknowledgment prose (`"Understood, reassessing..."`) rather than a JSON action.

**Required fixes:**

1. **Set a minimum `max_tokens`.** If `max_tokens` is too low, the model truncates mid-JSON. Minimum 256 for any decision tick.

2. **Re-prompt once on parse failure before erroring.** Do not propagate a parse failure directly — retry with an explicit JSON reminder:

```typescript
async function decide(agentState: AgentState): Promise<Action> {
  let raw = await llm.complete(agentState);

  if (!isValidJSON(raw)) {
    // Single retry with explicit format reminder
    raw = await llm.complete({
      ...agentState,
      systemOverride: 'Your previous response was not valid JSON. Respond only with a valid JSON action object. No prose.',
    });
  }

  if (!isValidJSON(raw)) {
    // Log raw output for diagnostics, then fall back to a safe default
    logger.error('LLM parse failure after retry', { raw });
    return { intent: 'hold' };  // safe no-op
  }

  return JSON.parse(raw);
}
```

3. **Log raw model output on failure.** The parse error alone is not enough to diagnose — you need the raw string. Add structured logging of the raw response whenever a parse failure occurs.

4. **Sanitize compaction output before injecting.** Strip any text that looks like a partial JSON action from the compaction summary before feeding it back into context.

---

### BUG 4: `scan.active` target_id must be a UUID

Observed in audit 2026-02-18: `Safety: scan.active target_id must be a UUID when provided` — safety gate blocked 1 action.

**Why this happens:** The model generates a `target_id` value by hallucinating an entity identifier (e.g., a ship name, a coordinate string, or a partial ID) rather than using a real UUID from the world state. It has no way to know valid UUIDs unless they appear in its context.

**Required fixes:**

1. **Only pass `target_id` when you have a real UUID from world state.** If no valid target entity is in context, omit the field entirely — `scan.active` without a `target_id` is a valid area scan.

2. **Enumerate valid target UUIDs in the actionable state.** If you want the model to scan specific entities, include their UUIDs explicitly in the context so the model can reference them:

```
NEARBY ENTITIES
- id: "fdcf66d3-b49e-48a4-8c80-d38e7749a1c9"  type: ship  name: To'vah
- id: "a3e2c1d4-..."                           type: station  name: Port Veluna
```

3. **Validate UUID format before execution.** The safety gate catches this, but the action is wasted. Add client-side validation: if `target_id` is present and not a valid UUID v4, strip it before submitting.

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

**Why structure matters:** LLMs attend to text in proportion to its salience and position. A constraint buried in a paragraph is weighted less than the same constraint as a standalone bullet point. Labeled sections (`BEHAVIORAL RULES`, `KNOWLEDGE LIMITS`) also give the model clear retrieval anchors — it can locate and apply the relevant section when generating a response, rather than trying to extract rules from narrative prose. Fewer tokens for the same instruction density means faster inference and more room for world context.

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

**Why it matters:** The heavy model (14B parameters) processes and generates tokens at approximately half the speed of the light model (8B) on the same hardware. For structured decisions — navigation, scanning, routine actions — the output is constrained to a JSON schema. Model size affects prose quality and reasoning depth, not JSON accuracy. Routing navigation decisions to `heavy` doubles latency for no observable quality difference, and fills heavy worker queues while light workers sit idle.

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
- Any structured action output (navigation, scanning, combat targeting)

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

**Why these specific thresholds:** The 6,000-token threshold comes from observed data — median prompt token usage in audits was 5,242–6,175 tokens. This is the point at which prompt size starts meaningfully affecting KV cache utilization and slowing inference. The 100-entry count threshold is a safety net for cases where individual entries are small; at ~60 bytes average per entry from audit data, 100 entries approximates the same 6K-token target. Both protect the same thing from two different angles — use whichever fires first.

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

**Why the model won't complete sequences on its own:** Each tick the model sees the current state and picks the best action for that state. It does not see its prior decisions. A two-step sequence like `set_destination → engage` requires the model to remember step one was already done — but it can't, because last tick's output is not fed back as input. The only reliable approach is to track sequence state in the loop and enforce follow-up actions in code (see BUG 1 fix above). The same principle applies to combat, docking, and warp sequences below.

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

### Model returning non-JSON or empty intent
**Symptom:** `Failed to parse LLM response`, empty intent `""` blocked by safety gate.
**Fix:** See **BUG 3** at the top of this document.

### Hallucinated `target_id` on `scan.active`
**Symptom:** `Safety: scan.active target_id must be a UUID when provided`.
**Fix:** See **BUG 4** at the top of this document.

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
- [ ] Stuck detection covers all intents via a single generic sliding-window check, not per-intent handlers
- [ ] JSON parse failure handled: retry with format reminder, fallback to `hold`, raw output logged
- [ ] `scan.active` only passes `target_id` when a valid UUID is available in world state context
- [ ] `max_tokens` set to minimum 256 on all decision ticks
- [ ] P50 latency measured under expected load and within target for chosen model tier
