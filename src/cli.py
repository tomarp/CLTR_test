import argparse
from pathlib import Path
from .pipeline import run_batch
from .interactive_report import build_one_session_html, build_combined

def main():
    parser = argparse.ArgumentParser(description="CLTR Pipeline CLI")
    parser.add_argument("--sessions-root", type=Path, help="Path to sessions root")
    parser.add_argument("--outdir", type=Path, help="Path to output directory")
    parser.add_argument("--timeline-csv", type=Path, help="Path to timeline CSV")
    parser.add_argument("--timeline-tz", type=str, default="Europe/Paris", help="Timeline timezone")
    parser.add_argument("--interactive", action="store_true", help="Generate interactive HTML reports")

    args = parser.parse_args()

    if args.sessions_root and args.outdir and args.timeline_csv:
        # Run batch pipeline (PDFs and CSVs)
        run_batch(args.sessions_root, args.outdir, args.timeline_csv, args.timeline_tz)

        # Optionally run interactive reports
        if args.interactive:
            from .io import find_session_dirs
            session_dirs = find_session_dirs(args.outdir) # Results dirs
            for sd in session_dirs:
                build_one_session_html(sd, sd / f"{sd.name}_interactive_report.html")
            build_combined(args.outdir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
