#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import logging
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    files = sorted(p for p in data_dir.rglob("*.*") if p.is_file())
    return [f for f in files if f.suffix.lower() in {".json", ".csv"} and f.name.lower() != "locations.json"]


def _load_transactions_from_file(dataset_file: Path) -> list[dict[str, Any]]:
    if dataset_file.suffix.lower() == ".csv":
        with dataset_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    payload = load_json_file(dataset_file)
    return _extract_transactions(payload)


def _dataset_root_for_file(dataset_file: Path, data_dir: Path) -> Path:
    try:
        relative_path = dataset_file.relative_to(data_dir)
    except ValueError:
        return dataset_file.parent

    if not relative_path.parts:
        return data_dir

    return data_dir / relative_path.parts[0]


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

    # Log environment check
    import os
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.warning("❌ OPENROUTER_API_KEY is NOT set! LLM calls will fail.")
    else:
        logger.info(f"✓ OPENROUTER_API_KEY is set (first 10 chars: {api_key[:10]}...)")
    
    langfuse_host = os.getenv("LANGFUSE_HOST", "")
    logger.info(f"✓ LANGFUSE_HOST={langfuse_host}")
    
    timeout_seconds = os.getenv("TRANSACTION_TIMEOUT_SECONDS", "60")
    logger.info(f"✓ TRANSACTION_TIMEOUT_SECONDS={timeout_seconds}")

    parser = argparse.ArgumentParser(description="Wealth-Guardian MAS fraud detector")
    parser.add_argument("--data-dir", default="data", help="Directory containing challenge JSON datasets")
    parser.add_argument("--output", default="output.txt", help="Output file with one fraud ID per line")
    parser.add_argument("--team-name", default="REPLY-MIRROR", help="Team name used in Langfuse session_id")
    parser.add_argument("--fraud-threshold", type=float, default=0.65, help="Minimum confidence to mark fraud")
    parser.add_argument("--max-transactions", type=int, default=0, help="Optional cap for quick smoke tests")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    logger.info(f"Starting fraud detection with max_transactions={args.max_transactions} on {data_dir}")

    fraud_ids: list[str] = []
    forensic_memory = ForensicSharedMemory.instance()
    recurring_tracker = RecurringPatternTracker()
    mas = WealthGuardianMAS(team_name=args.team_name)
    processed_transactions = 0

    for dataset_file in _iter_dataset_files(data_dir):
        transactions = _load_transactions_from_file(dataset_file)

        for tx in transactions:
            if args.max_transactions > 0 and processed_transactions >= args.max_transactions:
                break

            tx_id = detect_transaction_id(tx)
            if not tx_id:
                continue

            processed_transactions += 1

            analysis_signals, tier = _analyze_transaction_signals(
                transaction=tx,
                dataset_root=_dataset_root_for_file(dataset_file, data_dir),
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

        if args.max_transactions > 0 and processed_transactions >= args.max_transactions:
            break

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


if __name__ == "__main__":
    run()
