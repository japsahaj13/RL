"""
Cleanup Results Script

This script cleans up the results directory, keeping only the most recent run in each category.
"""

import os
import shutil
import glob
from datetime import datetime

def cleanup_results_folder(keep_latest=1, backup=True):
    """
    Clean up the results directory, keeping only the most recent runs.
    
    Args:
        keep_latest: Number of latest runs to keep in each directory
        backup: Whether to backup removed runs
    """
    base_dir = "results"
    
    # Create backup directory if needed
    if backup:
        backup_dir = os.path.join(base_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)
    
    # Identify subdirectories (categories)
    categories = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and not item.startswith(".") and item != "backup":
            categories.append(item)
    
    total_removed = 0
    total_kept = 0
    
    print(f"Cleaning up results directory, keeping {keep_latest} most recent run(s) per category")
    print("-" * 80)
    
    # Process each category
    for category in categories:
        category_dir = os.path.join(base_dir, category)
        
        # Get all run directories (e.g., dqn_20250430_230104)
        run_dirs = []
        for item in os.listdir(category_dir):
            item_path = os.path.join(category_dir, item)
            if os.path.isdir(item_path) and item.startswith("dqn_"):
                run_dirs.append(item)
        
        # Sort by timestamp (most recent first)
        run_dirs.sort(reverse=True)
        
        # Keep only the most recent runs
        to_keep = run_dirs[:keep_latest]
        to_remove = run_dirs[keep_latest:]
        
        print(f"\nCategory: {category}")
        print(f"  Total runs: {len(run_dirs)}")
        print(f"  Keeping: {len(to_keep)}")
        print(f"  Removing: {len(to_remove)}")
        
        # Process runs to remove
        for run_dir in to_remove:
            src_path = os.path.join(category_dir, run_dir)
            
            if backup:
                # Create category backup directory
                cat_backup_dir = os.path.join(backup_dir, category)
                os.makedirs(cat_backup_dir, exist_ok=True)
                
                # Move to backup
                dest_path = os.path.join(cat_backup_dir, run_dir)
                shutil.move(src_path, dest_path)
                print(f"  - Backed up: {run_dir}")
            else:
                # Remove directory
                shutil.rmtree(src_path)
                print(f"  - Removed: {run_dir}")
                
            total_removed += 1
        
        total_kept += len(to_keep)
    
    print("\n" + "-" * 80)
    print(f"Cleanup complete: Kept {total_kept} runs, Removed/Backed up {total_removed} runs")

if __name__ == "__main__":
    cleanup_results_folder(keep_latest=1, backup=True)
    
    # Also remove the default_backup directory
    default_backup_dir = os.path.join("results", "default_backup")
    if os.path.exists(default_backup_dir):
        print(f"\nRemoving temporary backup directory: {default_backup_dir}")
        shutil.rmtree(default_backup_dir)
        print("Temporary backup directory removed") 