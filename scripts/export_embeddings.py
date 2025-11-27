#!/usr/bin/env python3
"""
CLI script for exporting BirdNET V3 embeddings from audio files.

Usage:
    # Export embeddings for a single file
    python scripts/export_embeddings.py --wav /path/to/audio.wav --out /path/to/output.npz

    # Export embeddings for all WAVs in a directory
    python scripts/export_embeddings.py --wav_dir /data/anuraset/wavs --out_dir /data/embeddings

    # Export with custom chunk length and overlap
    python scripts/export_embeddings.py --wav_dir /data/wavs --out_dir /data/embeddings \
        --chunk_length 1.0 --overlap 0.0 --batch_size 64
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from birdnetv3 import embed_file, embed_directory


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export BirdNET V3 embeddings from audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input options (mutually exclusive groups)
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--wav", type=str, default=None,
        help="Path to single audio file to embed",
    )
    input_group.add_argument(
        "--wav_dir", type=str, default=None,
        help="Directory containing audio files to embed",
    )
    input_group.add_argument(
        "--list_file", type=str, default=None,
        help="Text file with paths to audio files (one per line)",
    )
    input_group.add_argument(
        "--glob", type=str, default="**/*.wav",
        help="Glob pattern for finding audio files (default: **/*.wav)",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--out", type=str, default=None,
        help="Output path for single file (default: <wav>.embeddings.npz)",
    )
    output_group.add_argument(
        "--out_dir", type=str, default="embeddings",
        help="Output directory for batch processing (default: embeddings)",
    )
    output_group.add_argument(
        "--no_preserve_structure", action="store_true",
        help="Don't preserve subdirectory structure in output",
    )
    
    # Processing options
    proc_group = parser.add_argument_group("Processing")
    proc_group.add_argument(
        "--model", type=str, default=None,
        help="Path to BirdNET V3 TorchScript model file",
    )
    proc_group.add_argument(
        "--chunk_length", type=float, default=1.0,
        help="Chunk length in seconds (default: 1.0)",
    )
    proc_group.add_argument(
        "--overlap", type=float, default=0.0,
        help="Overlap between chunks in seconds (default: 0.0)",
    )
    proc_group.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for inference (default: 64)",
    )
    proc_group.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda"],
        help="Device for inference (default: auto-detect)",
    )
    
    # Other options
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Validate input arguments
    has_single = args.wav is not None
    has_batch = args.wav_dir is not None or args.list_file is not None
    
    if not has_single and not has_batch:
        parser.error("Must specify either --wav for single file or --wav_dir/--list_file for batch processing")
    
    if has_single and has_batch:
        parser.error("Cannot specify both --wav and --wav_dir/--list_file")
    
    try:
        if has_single:
            # Single file mode
            wav_path = Path(args.wav)
            if not wav_path.exists():
                logger.error(f"Audio file not found: {wav_path}")
                return 1
            
            out_path = Path(args.out) if args.out else wav_path.with_suffix(".embeddings.npz")
            
            logger.info(f"Embedding single file: {wav_path}")
            embed_file(
                wav_path=wav_path,
                out_path=out_path,
                model_path=args.model,
                device=args.device,
                chunk_length=args.chunk_length,
                overlap=args.overlap,
                batch_size=args.batch_size,
            )
            logger.info(f"Saved embeddings to: {out_path}")
            
        else:
            # Batch mode
            logger.info("Starting batch embedding export")
            count = embed_directory(
                wav_dir=args.wav_dir,
                out_dir=args.out_dir,
                glob_pattern=args.glob,
                list_file=args.list_file,
                model_path=args.model,
                device=args.device,
                chunk_length=args.chunk_length,
                overlap=args.overlap,
                batch_size=args.batch_size,
                preserve_structure=not args.no_preserve_structure,
            )
            logger.info(f"Successfully embedded {count} files")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
