#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from reply_challenge.crew import WealthGuardianMAS
from reply_challenge.tools import (
    ForensicSharedMemory,
    RecurringPatternTracker,
    analyze_vishing_text,
    compute_tier,
    detect_amount,
    detect_audio_path,
    detect_coordinates,
    detect_text_payload,
    detect_timestamp,
    detect_transaction_id,
    detect_user_id,
    load_json_file,
    transcribe_audio,
)


def _extract_transactions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("transactions", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _iter_dataset_files(data_dir: Path) -> list[Path]:
    files = sorted(p for p in data_dir.rglob("*.json") if p.is_file())
    return [f for f in files if f.name.lower() != "locations.json"]


def _analyze_transaction_signals(
    transaction: dict[str, Any],
    dataset_root: Path,
    forensic_memory: ForensicSharedMemory,
    recurring_tracker: RecurringPatternTracker,
) -> tuple[dict[str, Any], int]:
    user_id = detect_user_id(transaction)
    amount = detect_amount(transaction)
    tx_text = detect_text_payload(transaction)

    recurring_flags = recurring_tracker.evaluate(transaction)
    coords = detect_coordinates(transaction)
    timestamp = detect_timestamp(transaction)
    geo_signal = forensic_memory.evaluate_impossible_travel(user_id=user_id, timestamp=timestamp, coords=coords)

    audio_text = transcribe_audio(detect_audio_path(transaction, dataset_root))
    cyber_text = "\n".join(chunk for chunk in (tx_text, audio_text) if chunk).strip()
    vishing = analyze_vishing_text(cyber_text)

    behavioral_anomaly = bool(geo_signal.get("flag") or vishing.get("is_suspicious"))
    anomaly_flags = {
        **recurring_flags,
        "any_behavioral_anomaly": behavioral_anomaly,
    }
    tier = compute_tier(transaction, anomaly_flags)

    if forensic_memory.is_user_compromised(user_id):
        anomaly_flags["known_compromised_user"] = True
        tier = 1

    analysis_signals = {
        "amount": amount,
        "anomaly_flags": anomaly_flags,
        "geo": geo_signal,
        "cyber": {
            "message_excerpt": cyber_text[:1000],
            "vishing": vishing,
        },
    }
    return analysis_signals, tier


def run() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Wealth-Guardian MAS fraud detector")
    parser.add_argument("--data-dir", default="data", help="Directory containing challenge JSON datasets")
    parser.add_argument("--output", default="output.txt", help="Output file with one fraud ID per line")
    parser.add_argument("--team-name", default="REPLY-MIRROR", help="Team name used in Langfuse session_id")
    parser.add_argument("--fraud-threshold", type=float, default=0.65, help="Minimum confidence to mark fraud")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    fraud_ids: list[str] = []
    forensic_memory = ForensicSharedMemory.instance()
    recurring_tracker = RecurringPatternTracker()
    mas = WealthGuardianMAS(team_name=args.team_name)

    for dataset_file in _iter_dataset_files(data_dir):
        payload = load_json_file(dataset_file)
        transactions = _extract_transactions(payload)

        for tx in transactions:
            tx_id = detect_transaction_id(tx)
            if not tx_id:
                continue

            analysis_signals, tier = _analyze_transaction_signals(
                transaction=tx,
                dataset_root=dataset_file.parent,
                forensic_memory=forensic_memory,
                recurring_tracker=recurring_tracker,
            )

            verdict = mas.investigate_transaction(tx, analysis_signals=analysis_signals, tier=tier)
            recurring_tracker.update(tx)

            if verdict.is_fraud and verdict.confidence >= args.fraud_threshold:
                fraud_ids.append(tx_id)
                forensic_memory.mark_user_compromised(detect_user_id(tx))

            for indicator in (tx.get("ip"), tx.get("iban"), tx.get("device_id"), tx.get("recipient_iban")):
                if indicator and verdict.is_fraud:
                    if "iban" in str(indicator).lower() or str(indicator).startswith("IT"):
                        forensic_memory.mark_indicator("iban", str(indicator))
                    elif "." in str(indicator):
                        forensic_memory.mark_indicator("ip", str(indicator))
                    else:
                        forensic_memory.mark_indicator("id", str(indicator))

    output_path = Path(args.output).resolve()
    output_path.write_text("\n".join(fraud_ids) + ("\n" if fraud_ids else ""), encoding="utf-8")


def train() -> None:
    raise NotImplementedError("Training mode is not used in the challenge solution.")


def replay() -> None:
    raise NotImplementedError("Replay mode is not used in the challenge solution.")


def test() -> None:
    raise NotImplementedError("Test mode is not used in the challenge solution.")


def run_with_trigger() -> None:
    run()
