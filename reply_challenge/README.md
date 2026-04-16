# The Wealth-Guardian MAS (Reply Mirror)

CrewAI hierarchical fraud detection system for AI Reply Challenge 2087.

## Core Features

- Dynamic Token Auctioneer for tier-based model routing.
- Forensic Shared Memory singleton across all datasets in one run.
- Multimodal vishing analysis with Whisper transcription.
- Hierarchical agent process with Devil's Advocate for high-value checks.
- Langfuse v3 tracing with `session_id={TEAM_NAME}-{ULID}`.

## Environment Variables

Set these variables in `.env`:

```bash
OPENROUTER_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://challenges.reply.com/langfuse
WHISPER_MODEL=base
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m reply_challenge.main --data-dir ./data --output ./output.txt --team-name REPLY-MIRROR
```

The output file contains one fraud transaction ID per line.

## Architecture

- Lead Orchestrator (The Eye): triage + final verdict synthesis.
- Geo-Spatial Investigator: impossible travel and geolocation anomalies.
- Cyber-Profiler: social engineering and vishing signals.
- The Devil's Advocate: challenge high-value fraud conclusions.
