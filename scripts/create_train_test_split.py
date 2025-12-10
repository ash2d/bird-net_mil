#!/usr/bin/env python3
"""
Create train/test split for bird-net embeddings.

The test set includes:
1. One complete site (the one with fewest recordings)
2. Last 5% of recordings from other sites (chronologically)

Outputs two text files listing paths to embeddings for train and test sets.

Usage:
    python scripts/create_train_test_split.py --emb_dir /data/embeddings \
        --train_out train.txt --test_out test.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def extract_site_from_path(npz_path: Path) -> str:
    """
    Extract site/monitoring location from embedding path.
    
    Assumes structure: .../SITE_NAME/filename.embeddings.npz
    
    Args:
        npz_path: Path to embedding file.
        
    Returns:
        Site name (parent directory name).
    """
    return npz_path.parent.name


def group_embeddings_by_site(npz_paths: List[Path]) -> Dict[str, List[Path]]:
    """
    Group embedding files by site.
    
    Args:
        npz_paths: List of paths to embedding files.
        
    Returns:
        Dictionary mapping site name to list of embedding paths.
    """
    site_groups = defaultdict(list)
    
    for npz_path in npz_paths:
        site = extract_site_from_path(npz_path)
        site_groups[site].append(npz_path)
    
    return dict(site_groups)


def find_smallest_site(site_groups: Dict[str, List[Path]]) -> str:
    """
    Find the site with the fewest recordings.
    
    Args:
        site_groups: Dictionary mapping site to list of paths.
        
    Returns:
        Name of site with fewest recordings.
    """
    return min(site_groups.keys(), key=lambda site: len(site_groups[site]))


def split_site_chronologically(
    paths: List[Path],
    test_fraction: float = 0.05,
) -> Tuple[List[Path], List[Path]]:
    """
    Split site recordings chronologically.
    
    Sorts paths alphabetically (assuming chronological ordering in filenames)
    and takes the last test_fraction for testing.
    
    Args:
        paths: List of paths for a single site.
        test_fraction: Fraction of recordings to use for testing (default: 0.05).
        
    Returns:
        Tuple of (train_paths, test_paths).
    """
    # Sort paths alphabetically (assumes filenames are chronologically ordered)
    sorted_paths = sorted(paths)
    
    # Calculate split point
    n_total = len(sorted_paths)
    n_test = max(1, int(n_total * test_fraction))  # At least 1 for test
    
    # Split: last n_test for testing, rest for training
    train_paths = sorted_paths[:-n_test]
    test_paths = sorted_paths[-n_test:]
    
    return train_paths, test_paths


def create_train_test_split(
    emb_dir: Path,
    test_fraction: float = 0.05,
) -> Tuple[List[Path], List[Path]]:
    """
    Create train/test split of embedding files.
    
    Test set includes:
    - All files from the site with fewest recordings
    - Last test_fraction (default 5%) of files from other sites
    
    Args:
        emb_dir: Root directory containing embeddings.
        test_fraction: Fraction of recordings from non-test sites (default: 0.05).
        
    Returns:
        Tuple of (train_paths, test_paths).
    """
    logger = logging.getLogger(__name__)
    
    # Find all .embeddings.npz files
    npz_pattern = "**/*.embeddings.npz"
    npz_paths = sorted(emb_dir.glob(npz_pattern))
    
    if not npz_paths:
        raise FileNotFoundError(f"No .embeddings.npz files found in {emb_dir}")
    
    logger.info(f"Found {len(npz_paths)} embedding files")
    
    # Group by site
    site_groups = group_embeddings_by_site(npz_paths)
    logger.info(f"Found {len(site_groups)} sites")
    
    for site, paths in site_groups.items():
        logger.info(f"  {site}: {len(paths)} files")
    
    # Find site with fewest recordings
    test_site = find_smallest_site(site_groups)
    logger.info(f"Site with fewest recordings: {test_site} ({len(site_groups[test_site])} files)")
    
    # Initialize train and test sets
    train_paths = []
    test_paths = []
    
    # Process each site
    for site, paths in site_groups.items():
        if site == test_site:
            # All files from smallest site go to test
            test_paths.extend(paths)
            logger.info(f"Added all {len(paths)} files from {site} to test set")
        else:
            # Split chronologically
            site_train, site_test = split_site_chronologically(paths, test_fraction)
            train_paths.extend(site_train)
            test_paths.extend(site_test)
            logger.info(
                f"Split {site}: {len(site_train)} train, {len(site_test)} test "
                f"({len(site_test)/len(paths)*100:.1f}%)"
            )
    
    logger.info(f"\nFinal split: {len(train_paths)} train, {len(test_paths)} test")
    logger.info(f"Test percentage: {len(test_paths)/len(npz_paths)*100:.1f}%")
    
    return train_paths, test_paths


def write_path_list(paths: List[Path], output_file: Path) -> None:
    """
    Write list of paths to text file.
    
    Args:
        paths: List of paths to write.
        output_file: Output text file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for path in sorted(paths):
            f.write(f"{path}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create train/test split for embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--emb_dir", type=str, required=True,
        help="Directory containing .embeddings.npz files",
    )
    parser.add_argument(
        "--train_out", type=str, default="train.txt",
        help="Output file for training paths (default: train.txt)",
    )
    parser.add_argument(
        "--test_out", type=str, default="test.txt",
        help="Output file for test paths (default: test.txt)",
    )
    parser.add_argument(
        "--test_fraction", type=float, default=0.05,
        help="Fraction of non-test-site data for testing (default: 0.05)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    try:
        emb_dir = Path(args.emb_dir)
        if not emb_dir.exists():
            logger.error(f"Directory not found: {emb_dir}")
            return 1
        
        # Create split
        logger.info(f"Creating train/test split from {emb_dir}")
        train_paths, test_paths = create_train_test_split(emb_dir, args.test_fraction)
        
        # Write output files
        train_out = Path(args.train_out)
        test_out = Path(args.test_out)
        
        write_path_list(train_paths, train_out)
        logger.info(f"Wrote {len(train_paths)} training paths to {train_out}")
        
        write_path_list(test_paths, test_out)
        logger.info(f"Wrote {len(test_paths)} test paths to {test_out}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
