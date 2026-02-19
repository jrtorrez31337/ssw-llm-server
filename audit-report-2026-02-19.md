# NPC Agent Audit Report

_Generated: 2026-02-19 02:53 UTC_

## Session Overview

- **Agents**: 2

- **Total decisions**: 380

- **Total memories**: 692

- **Total conversations**: 5


### Agent Summary

| Agent ID | Character | Ship | Ticks | Duration | Decisions |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `fighter-agent-01` | Lt Klank | To'vah | 236 | 5.3h | 235 |
| `patrol` | Space X-1 | Space X-1 | 525 | 1.1h | 145 |


### Data Sources

| Agent | File | Records | Size |
| :--- | :--- | ---: | ---: |
| `fighter-agent-01` | `npc-agent-fighter-agent-01.jsonl` | 235 | 7.0MB |
| `fighter-agent-01` | `memory-fighter-agent-01.jsonl` | 280 | 129.5KB |
| `fighter-agent-01` | `conversations-fighter-agent-01.jsonl` | 5 | 1.4KB |
| `fighter-agent-01` | `state-fighter-agent-01.json` | 1 | 124B |
| `patrol` | `npc-agent-patrol.jsonl` | 145 | 5.3MB |
| `patrol` | `memory-patrol.jsonl` | 412 | 245.6KB |
| `patrol` | `state-patrol.json` | 1 | 124B |

## Intent Distribution

### Fleet-Wide

```
  course.set_destination  ██████████████████████████████ 204
  scan.active             ██████████████ 96
  scan.passive            ██████ 44
  navigate.set_course     █████ 32
  course.cancel            3
  combat.set_target        1

```

### Per-Agent Breakdown

| Intent | `fighter-agent-01` | `patrol` | Total |
| :--- | ---: | ---: | ---: |
| `course.set_destination` | 204 | - | 204 |
| `scan.active` | 26 | 70 | 96 |
| `scan.passive` | - | 44 | 44 |
| `navigate.set_course` | 1 | 31 | 32 |
| `course.cancel` | 3 | - | 3 |
| `combat.set_target` | 1 | - | 1 |


### Never-Used Intents

These known action types were never selected during this session:

- `chat.hail`
- `chat.send_message`
- `combat.clear_target`
- `combat.fire_weapon`
- `combat.power_distribution`
- `course.set_coordinates`
- `flee`
- `hold`
- `mining.extract`
- `navigate.dock`
- `navigate.jump`
- `navigate.undock`
- `orbit.enter`
- `orbit.exit`
- `remember`
- `station.refuel`
- `station.repair`
- `travel.cancel`
- `warp.choice`
- `warp.compute`
- `warp.drop`
- `warp.engage`
- `warp.spool`

## LLM Performance

### LLM Latency

- **Decisions**: 380

- **Min**: 8.2s

- **P50**: 44.8s

- **P95**: 2.0m

- **P99**: 3.8m

- **Max**: 4.4m

- **Mean**: 52.4s


### Action Execution Latency

- **P50**: 35ms

- **P95**: 97ms

- **Max**: 183ms


### Prompt Token Usage

- **Min**: 4,272

- **P50**: 6,172

- **P95**: 8,012

- **Max**: 8,468

- **Mean**: 6,269

- **Budget**: 24,000

- **Over budget**: 0x

- **Prune failed**: 0x


### Context Layer Inclusion

| Layer | Included | Total | Rate |
| :--- | ---: | ---: | ---: |
| `actionable_state` | 380 | 380 | 100% |
| `actions` | 380 | 380 | 100% |
| `client_stores` | 380 | 380 | 100% |
| `guidance` | 289 | 289 | 100% |
| `memory` | 288 | 288 | 100% |
| `output_contract` | 380 | 380 | 100% |
| `pending_requests` | 5 | 5 | 100% |
| `recent_actions` | 378 | 378 | 100% |
| `recent_events` | 82 | 82 | 100% |
| `situation` | 380 | 380 | 100% |
| `system` | 380 | 380 | 100% |


## Execution Health

- **Total decisions**: 380

- **Successful**: 380 (100.0%)

- **Failed**: 0 (0.0%)

- **Blocked (not allowed)**: 0


## Memory System

- **Total entries**: 692

- **Compaction events**: 0

- **Reflection events**: 31


### Entries Per Agent

| Agent | Memories |
| :--- | ---: |
| `fighter-agent-01` | 280 |
| `patrol` | 412 |


### Memory Types

```
  encounter    ██████████████████████████████ 413
  observation  ██████████████ 196
  reflection   ██████ 83

```

### Confidence Distribution

- **Min**: 0.200

- **P50**: 0.720

- **P95**: 0.742

- **Max**: 0.800


### Salience Distribution

- **Min**: 0.200

- **P50**: 0.580

- **P95**: 0.750

- **Max**: 0.750


### Top Tags

| Tag | Count |
| :--- | ---: |
| `decision` | 447 |
| `rationale` | 446 |
| `course.set_destination` | 227 |
| `generic_event` | 184 |
| `scan.active` | 117 |
| `player.inferred.game.travel.started` | 92 |
| `player.inferred.game.travel.completed` | 92 |
| `reflection` | 89 |
| `scan.passive` | 57 |
| `memory_compaction` | 50 |
| `navigate.set_course` | 37 |
| `game.chat.message` | 12 |
| `chat` | 12 |
| `comms` | 12 |
| `ent:fdcf66d3-b49e-48a4-8c80-d38e7749a1c9` | 12 |
| `distress` | 7 |
| `course.cancel` | 6 |
| `combat.set_target` | 1 |


## Chat Activity

- **Total exchanges**: 5

- **Unique players**: 1

- **Unique rooms**: 1


### Exchanges Per Agent

| Agent | Exchanges |
| :--- | ---: |
| `fighter-agent-01` | 5 |
| `patrol` | 0 |


### Conversations By Player

| Player | Messages |
| :--- | ---: |
| Lt Slurr | 5 |


### Agent Reply Length

- **Min chars**: 3

- **P50 chars**: 3

- **Max chars**: 61

- **P50 words**: 1

- **Max words**: 11


### Repeated Replies

> Replies that appear 3+ times may indicate stuck behavior.

| Reply | Count |
| :--- | ---: |
| `In.` | 4 |


## Safety & Validation

### Schema Validation

- **Valid**: 380 (100.0%)

- **Invalid**: 0 (0.0%)


### Safety Gate

- **Passed**: 380 (100.0%)

- **Failed**: 0 (0.0%)


## Behavioral Patterns

### Top Intent Bigrams

| Sequence | Count |
| :--- | ---: |
| `course.set_destination` -> `course.set_destination` | 173 |
| `scan.active` -> `scan.active` | 43 |
| `scan.active` -> `course.set_destination` | 26 |
| `course.set_destination` -> `scan.active` | 25 |
| `scan.passive` -> `navigate.set_course` | 25 |
| `scan.active` -> `scan.passive` | 20 |
| `navigate.set_course` -> `scan.passive` | 17 |
| `navigate.set_course` -> `scan.active` | 14 |
| `scan.passive` -> `scan.active` | 12 |
| `scan.passive` -> `scan.passive` | 7 |
| `scan.active` -> `navigate.set_course` | 6 |
| `course.set_destination` -> `course.cancel` | 3 |
| `course.cancel` -> `course.set_destination` | 3 |
| `course.set_destination` -> `combat.set_target` | 1 |
| `combat.set_target` -> `course.set_destination` | 1 |


### Top Intent Trigrams

| Sequence | Count |
| :--- | ---: |
| `course.set_destination` -> `course.set_destination` -> `course.set_destination` | 142 |
| `scan.active` -> `scan.active` -> `scan.active` | 29 |
| `scan.active` -> `course.set_destination` -> `course.set_destination` | 26 |
| `course.set_destination` -> `course.set_destination` -> `scan.active` | 25 |
| `course.set_destination` -> `scan.active` -> `course.set_destination` | 25 |
| `scan.passive` -> `navigate.set_course` -> `scan.passive` | 15 |
| `scan.active` -> `scan.passive` -> `navigate.set_course` | 12 |
| `scan.passive` -> `navigate.set_course` -> `scan.active` | 10 |
| `scan.active` -> `scan.active` -> `scan.passive` | 10 |
| `navigate.set_course` -> `scan.active` -> `scan.active` | 9 |


### Stuck Detection

> Same intent repeated 3+ times consecutively.

- `fighter-agent-01`: **`course.set_destination`** x4 (ticks 2-5)
- `fighter-agent-01`: **`course.set_destination`** x11 (ticks 7-17)
- `fighter-agent-01`: **`course.set_destination`** x8 (ticks 19-26)
- `fighter-agent-01`: **`course.set_destination`** x9 (ticks 28-36)
- `fighter-agent-01`: **`course.set_destination`** x6 (ticks 38-43)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 45-49)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 51-55)
- `fighter-agent-01`: **`course.set_destination`** x7 (ticks 57-63)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 65-69)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 71-75)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 77-81)
- `fighter-agent-01`: **`course.set_destination`** x6 (ticks 83-88)
- `fighter-agent-01`: **`course.set_destination`** x7 (ticks 90-96)
- `fighter-agent-01`: **`course.set_destination`** x8 (ticks 98-105)
- `fighter-agent-01`: **`course.set_destination`** x7 (ticks 107-113)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 115-119)
- `fighter-agent-01`: **`course.set_destination`** x6 (ticks 121-126)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 128-132)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 134-138)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 140-144)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 146-150)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 152-156)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 158-162)
- `fighter-agent-01`: **`course.set_destination`** x8 (ticks 164-171)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 173-177)
- `fighter-agent-01`: **`course.set_destination`** x12 (ticks 179-190)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 192-196)
- `fighter-agent-01`: **`course.set_destination`** x22 (ticks 198-219)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 221-225)
- `fighter-agent-01`: **`course.set_destination`** x5 (ticks 227-231)
- `fighter-agent-01`: **`course.set_destination`** x3 (ticks 233-235)
- `patrol`: **`scan.active`** x5 (ticks 26-30)
- `patrol`: **`scan.active`** x3 (ticks 36-38)
- `patrol`: **`scan.active`** x5 (ticks 43-47)
- `patrol`: **`scan.active`** x5 (ticks 54-58)
- `patrol`: **`scan.active`** x5 (ticks 64-68)
- `patrol`: **`scan.active`** x3 (ticks 86-88)
- `patrol`: **`scan.active`** x5 (ticks 94-98)
- `patrol`: **`scan.active`** x5 (ticks 103-107)
- `patrol`: **`scan.active`** x3 (ticks 109-111)
- `patrol`: **`scan.active`** x4 (ticks 116-119)
- `patrol`: **`scan.active`** x5 (ticks 123-127)
- `patrol`: **`scan.active`** x5 (ticks 132-136)

### Recovery Events

- **Stuck overrides (re-prompt)**: 42


### Flight State Distribution

```
  idle      ██████████████████████████████ 289
  en_route  █████████ 91

```

| State | `fighter-agent-01` | `patrol` | Total |
| :--- | ---: | ---: | ---: |
| `idle` | 144 | 145 | 289 |
| `en_route` | 91 | - | 91 |


### Sector Movement

- **`fighter-agent-01` unique sectors**: 8

- **`fighter-agent-01` sector transitions**: 7

  Route: `0.0.-1:0.0.193` -> `0.0.0:5.14.189` -> `1.0.0:13.23.177` -> `2.0.0:5.17.187` -> `0.0.1:13.26.175` -> `-1.0.0:5.40.171` -> `0.0.-1:5.40.171` -> `0.1.0:13.49.159`

- **`patrol` unique sectors**: 74

- **`patrol` sector transitions**: 74

  Route: `-1.0.0:125.168.78` -> `0.0.0:111.162.88` -> `0.0.1:123.154.94` -> `1.0.0:113.168.90` -> `0.0.0:113.168.90` -> `2.0.0:125.160.96` -> `0.0.1:125.160.96` -> `-1.0.0:133.169.84` -> `-2.0.0:119.163.94` -> `-1.-1.0:127.172.82` ... (+64 more)


---
_End of report._
