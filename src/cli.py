"""
CLI Tool for HCP Merge Assistant

Commands:
    analyze     Parse file and run match analysis
    merge       Execute merges from a previous analysis
    health      Check Reltio and LLM connectivity
    report      Generate report from analysis session

Usage:
    python -m src.cli analyze --file hcp_records.csv
    python -m src.cli analyze --file hcp_records.csv --entity-type HCP --output results.json
    python -m src.cli merge --file results.json --dry-run
    python -m src.cli merge --file results.json --confirm
    python -m src.cli health
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.merge_assistant import AssistantConfig, MergeAssistant  # noqa: E402


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


def get_config_from_args(args) -> AssistantConfig:
    """Build configuration from CLI args and environment"""
    return AssistantConfig(
        reltio_client_id=args.client_id or os.getenv("RELTIO_CLIENT_ID", ""),
        reltio_client_secret=args.client_secret or os.getenv("RELTIO_CLIENT_SECRET", ""),
        reltio_tenant_id=args.tenant_id or os.getenv("RELTIO_TENANT_ID", ""),
        reltio_environment=args.environment or os.getenv("RELTIO_ENVIRONMENT", "dev"),
        llm_api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        llm_provider="anthropic" if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY") else "openai",
        use_llm=not args.no_llm if hasattr(args, 'no_llm') else True,
        entity_type=getattr(args, 'entity_type', 'HCP'),
        max_concurrent_requests=getattr(args, 'concurrency', 10),
        auto_merge_threshold=getattr(args, 'auto_threshold', 95.0),
        review_threshold=getattr(args, 'review_threshold', 70.0)
    )


def print_banner():
    """Print application banner"""
    print("""
 ================================================
   HCP Merge Assistant for Reltio MDM
   Intelligent Bulk Match & Merge
 ================================================
""")


def print_progress(completed: int, total: int):
    """Print a progress bar to the terminal"""
    width = 40
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = completed / total * 100
    print(f"\r  [{bar}] {pct:.0f}% ({completed}/{total})", end="", flush=True)
    if completed == total:
        print()  # Newline when done


def cmd_health(args):
    """Handle 'health' command"""
    print_banner()
    print("Checking connections...\n")

    config = get_config_from_args(args)
    assistant = MergeAssistant(config)
    health = assistant.run_health_check()

    # Reltio status
    reltio = health["reltio"]
    if reltio["status"] == "healthy":
        print("  Reltio:  CONNECTED")
        print(f"  Tenant:  {config.reltio_tenant_id}")
        print(f"  Env:     {config.reltio_environment}")
    else:
        print(f"  Reltio:  FAILED - {reltio.get('error', 'Unknown')}")

    # LLM status
    llm = health["llm"]
    if llm["status"] == "configured":
        print(f"  LLM:     {llm['provider'].upper()} configured")
    elif llm["status"] == "not_configured":
        print("  LLM:     Not configured (scoring only, no AI analysis)")
    else:
        print(f"  LLM:     ERROR - {llm.get('error', 'Unknown')}")

    print()
    return 0 if reltio["status"] == "healthy" else 1


def cmd_analyze(args):
    """Handle 'analyze' command"""
    print_banner()

    file_path = args.file
    if not Path(file_path).exists():
        print(f"  ERROR: File not found: {file_path}")
        return 1

    config = get_config_from_args(args)
    assistant = MergeAssistant(config)

    # Step 1: Parse file
    print(f"  Step 1/3: Parsing file '{Path(file_path).name}'...")
    try:
        session = assistant.create_session(file_path)
    except Exception as e:
        print(f"  ERROR: Failed to parse file: {e}")
        return 1

    parsed = session.parsed_file
    print(f"    Records: {parsed.total_records} total, {parsed.valid_records} valid, {parsed.invalid_records} invalid")
    print(f"    Columns: {', '.join(parsed.detected_columns[:8])}{'...' if len(parsed.detected_columns) > 8 else ''}")
    print()

    # Step 2: Run match analysis
    print(f"  Step 2/3: Analyzing matches against Reltio ({config.entity_type})...")
    start_time = time.time()

    try:
        results = assistant.run_analysis(session, on_progress=print_progress)
    except Exception as e:
        print(f"\n  ERROR: Analysis failed: {e}")
        return 1

    duration = time.time() - start_time
    print(f"    Completed in {duration:.1f}s ({len(results)/max(duration,0.1):.1f} records/sec)")
    print()

    # Step 3: Summary
    print("  Step 3/3: Results Summary")
    summary = assistant.get_merge_summary(session)

    print(f"    With matches:      {summary['with_matches']}")
    print(f"    No matches:        {summary['no_matches']}")
    print(f"    Exact confidence:  {summary['by_confidence']['exact']}")
    print(f"    High confidence:   {summary['by_confidence']['high']}")
    print(f"    Medium confidence: {summary['by_confidence']['medium']}")
    print(f"    Low confidence:    {summary['by_confidence']['low']}")
    print()
    print(f"    Recommended merge: {summary['by_recommendation']['merge']}")
    print(f"    Needs review:      {summary['by_recommendation']['review']}")
    print(f"    No match:          {summary['by_recommendation']['no_match']}")
    print()

    # Auto-select if requested
    if args.auto_select:
        count = assistant.auto_select_merges(session, min_score=args.auto_threshold)
        print(f"  Auto-selected {count} merges above {args.auto_threshold}% threshold")
        print()

    # Output results
    output_path = args.output or f"analysis_{Path(file_path).stem}_{int(time.time())}.json"
    assistant.export_report(session, output_path, format="json")
    print(f"  Report saved to: {output_path}")

    # Also export CSV if requested
    if args.csv:
        csv_path = Path(output_path).with_suffix('.csv')
        assistant.export_report(session, str(csv_path), format="csv")
        print(f"  CSV saved to: {csv_path}")

    # Print individual records if verbose
    if args.verbose:
        print("\n  Detailed Results:")
        print("  " + "-" * 80)
        for i, result in enumerate(results[:20]):
            rec = result.input_record
            bm = result.best_match
            selected = "[X]" if i in session.selected_merges else "[ ]"

            input_label = " | ".join(f"{k}={v}" for k, v in list(rec.normalized_data.items())[:4])

            if bm:
                print(f"  {selected} Row {rec.row_number}: {input_label}")
                print(
                    f"        -> {bm.entity_label} ({bm.entity_uri})"
                    f" Score: {bm.match_score:.0f}% [{bm.confidence.value}]"
                )
            else:
                print(f"  {selected} Row {rec.row_number}: {input_label}")
                print("        -> No match found")

        if len(results) > 20:
            print(f"  ... and {len(results) - 20} more records (see report for full details)")

    print()
    return 0


def cmd_merge(args):
    """Handle 'merge' command"""
    print_banner()

    # Load analysis report
    report_path = args.file
    if not Path(report_path).exists():
        print(f"  ERROR: Report file not found: {report_path}")
        return 1

    with open(report_path, 'r') as f:
        report = json.load(f)

    # Extract merge pairs from report
    merge_pairs = []
    match_details = report.get("match_details", [])

    for detail in match_details:
        if not detail.get("selected_for_merge"):
            continue

        best = detail.get("best_match")
        input_data = detail.get("input_data", {})

        if best and input_data.get("EntityURI"):
            merge_pairs.append({
                "winner_uri": best["uri"],
                "loser_uri": input_data["EntityURI"],
                "row": detail["row"],
                "score": best["score"],
                "confidence": best["confidence"]
            })

    if not merge_pairs:
        print("  No merge operations found in report.")
        print("  Ensure records are selected (selected_for_merge = true) and have EntityURI identifiers.")
        return 1

    print(f"  Found {len(merge_pairs)} merge operations in report")
    print()

    # Show preview
    print("  Preview (first 10):")
    for pair in merge_pairs[:10]:
        print(f"    Row {pair['row']}: {pair['loser_uri']} -> {pair['winner_uri']} (Score: {pair['score']:.0f}%)")
    if len(merge_pairs) > 10:
        print(f"    ... and {len(merge_pairs) - 10} more")
    print()

    # Dry run check
    if args.dry_run:
        print("  DRY RUN MODE: Validating operations without executing merges")
    elif not args.confirm:
        # Ask for confirmation
        print(f"  WARNING: This will merge {len(merge_pairs)} entity pairs in Reltio.")
        print(f"  Environment: {os.getenv('RELTIO_ENVIRONMENT', 'dev')}")
        print()
        response = input("  Type 'yes' to proceed, or 'dry-run' to validate first: ").strip().lower()
        if response == "dry-run":
            args.dry_run = True
        elif response != "yes":
            print("  Cancelled.")
            return 0

    # Execute merges
    config = get_config_from_args(args)
    assistant = MergeAssistant(config)

    print(f"\n  {'Validating' if args.dry_run else 'Executing'} merges...")

    def on_merge_progress(completed, total, operation):
        print_progress(completed, total)

    try:
        result = asyncio.run(assistant.merge_entity_pairs(
            pairs=merge_pairs,
            on_progress=on_merge_progress,
            dry_run=args.dry_run
        ))
    except Exception as e:
        print(f"\n  ERROR: Merge execution failed: {e}")
        return 1

    print()
    print("  Results:")
    print(f"    Total:      {result.total_operations}")
    print(f"    Successful: {result.successful}")
    print(f"    Failed:     {result.failed}")
    print(f"    Duration:   {result.duration_seconds:.1f}s")

    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for error in result.errors[:5]:
            print(f"    - {error}")

    # Save merge report
    merge_report_path = args.merge_output or f"merge_result_{int(time.time())}.json"
    from src.core.merge_executor import MergeExecutor
    executor = MergeExecutor()
    executor.export_report(result, merge_report_path)
    print(f"\n  Merge report saved to: {merge_report_path}")

    print()
    return 0 if result.failed == 0 else 1


def cmd_report(args):
    """Handle 'report' command - convert JSON report to CSV"""
    report_path = args.file
    if not Path(report_path).exists():
        print(f"ERROR: File not found: {report_path}")
        return 1

    with open(report_path, 'r') as f:
        report = json.load(f)

    output = args.output or str(Path(report_path).with_suffix('.csv'))

    import csv
    details = report.get("match_details", [])
    if not details:
        print("No match details in report.")
        return 1

    rows = []
    for d in details:
        row = {"row": d["row"], "recommendation": d["recommendation"], "selected": d.get("selected_for_merge", False)}
        row.update(d.get("input_data", {}))
        bm = d.get("best_match")
        if bm:
            row["match_uri"] = bm["uri"]
            row["match_label"] = bm["label"]
            row["match_score"] = bm["score"]
            row["match_confidence"] = bm["confidence"]
            row["match_reasons"] = "; ".join(bm.get("reasons", []))
        rows.append(row)

    # Get all fieldnames
    fieldnames = list(dict.fromkeys(k for row in rows for k in row.keys()))

    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV report saved to: {output} ({len(rows)} records)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser"""
    parser = argparse.ArgumentParser(
        prog="hcp-merge-assistant",
        description="HCP Merge Assistant - Intelligent bulk match & merge for Reltio MDM"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Common Reltio args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--client-id", help="Reltio OAuth Client ID (or set RELTIO_CLIENT_ID)")
    common.add_argument("--client-secret", help="Reltio OAuth Client Secret (or set RELTIO_CLIENT_SECRET)")
    common.add_argument("--tenant-id", help="Reltio Tenant ID (or set RELTIO_TENANT_ID)")
    common.add_argument("--environment", default=None, help="Reltio environment: dev, test, prod")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Health command
    subparsers.add_parser("health", parents=[common], help="Check connectivity")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", parents=[common], help="Analyze HCP file for matches")
    analyze_parser.add_argument("--file", "-f", required=True, help="Input file (CSV, Excel, JSON)")
    analyze_parser.add_argument("--entity-type", default="HCP", help="Entity type (default: HCP)")
    analyze_parser.add_argument("--output", "-o", help="Output report path (default: auto-generated)")
    analyze_parser.add_argument("--csv", action="store_true", help="Also export CSV report")
    analyze_parser.add_argument("--no-llm", action="store_true", help="Disable LLM analysis")
    analyze_parser.add_argument("--auto-select", action="store_true", help="Auto-select high-confidence merges")
    analyze_parser.add_argument(
        "--auto-threshold", type=float, default=95.0, help="Auto-merge score threshold (default: 95)"
    )
    analyze_parser.add_argument(
        "--review-threshold", type=float, default=70.0, help="Review score threshold (default: 70)"
    )
    analyze_parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API requests (default: 10)")

    # Merge command
    merge_parser = subparsers.add_parser("merge", parents=[common], help="Execute merges from analysis report")
    merge_parser.add_argument("--file", "-f", required=True, help="Analysis report JSON file")
    merge_parser.add_argument("--dry-run", action="store_true", help="Validate without executing")
    merge_parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")
    merge_parser.add_argument("--merge-output", help="Merge result report path")
    merge_parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent merge operations")

    # Report command
    report_parser = subparsers.add_parser("report", help="Convert analysis JSON report to CSV")
    report_parser.add_argument("--file", "-f", required=True, help="Analysis report JSON file")
    report_parser.add_argument("--output", "-o", help="Output CSV path")

    return parser


def main():
    """Main entry point"""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    setup_logging(args.verbose)

    commands = {
        "health": cmd_health,
        "analyze": cmd_analyze,
        "merge": cmd_merge,
        "report": cmd_report,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
