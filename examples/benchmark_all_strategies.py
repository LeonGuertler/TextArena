"""
Benchmark script to compare all strategies.

Runs each strategy 5 times per instance and calculates:
- Average reward
- Standard deviation
- Min/Max rewards

Usage:
  uv run python D:\\TextArena\\examples\\benchmark_all_strategies.py --promised-lead-time 0 --directory D:\\TextArena\\examples\\initial_synthetic_demand_files\\case1_iid_normal
"""

import os
import sys
import subprocess
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from typing import Dict, List, Tuple

# Get the current Python executable
PYTHON_EXECUTABLE = sys.executable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Scripts to benchmark (all 4 strategies)
SCRIPTS = {
    "or": "examples/or_csv_demo.py",
    "llm": "examples/llm_csv_demo.py",
    "llm_to_or": "examples/llm_to_or_csv_demo.py",
    "or_to_llm": "examples/or_to_llm_csv_demo.py",
}

# Number of runs per script-instance combination
NUM_RUNS = 5

# Base directory
BASE_DIR = Path(__file__).parent.parent


def extract_reward_from_output(output: str) -> float:
    """Extract total reward from script output."""
    # Pattern to match: >>> Total Reward: $1234.56 <<< or $-123.45 <<<
    # or variations like "Total Reward (OR Baseline): $1234.56"
    # Note: Must handle negative numbers correctly!
    patterns = [
        r">>>\s*Total Reward[^:]*:\s*\$(-?\s*[\d,]+\.?\d*)\s*<<<",  # With >>> ... <<< (supports negative)
        r"Total Reward[^:]*:\s*\$(-?\s*[\d,]+\.?\d*)",  # Without >>> ... <<< (supports negative)
        r"VM Final Reward:\s*(-?\s*[\d,]+\.?\d*)",  # Fallback to VM Final Reward (supports negative)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            try:
                # Remove commas and whitespace from number string
                reward_str = match.group(1).replace(',', '').replace(' ', '')
                return float(reward_str)
            except ValueError:
                continue
    
    # If no match found, try to find any number after "Total Reward"
    lines = output.split('\n')
    for line in lines:
        if 'Total Reward' in line or 'total_reward' in line.lower():
            # Try to extract number (handle comma-separated numbers and negative signs)
            # Look for pattern: $ followed by optional minus, then number
            numbers = re.findall(r'\$(-?\s*[\d,]+\.?\d*)', line)
            if numbers:
                try:
                    # Take the last number and remove commas and whitespace
                    reward_str = numbers[-1].replace(',', '').replace(' ', '')
                    return float(reward_str)
                except ValueError:
                    continue
    
    return None


def run_script(script_path: str, run_num: int, script_name: str,
               promised_lead_time: int, instance_dir: str,
               max_periods: int = None, base_dir: str = None) -> Tuple[float, str, str]:
    """
    Run a script for a given instance and return the reward and output.
    
    Returns:
        (reward, error_message, output)
    """
    # Use provided paths or fall back to module-level constants
    if base_dir is None:
        base_dir = str(BASE_DIR)
    
    test_file = os.path.join(instance_dir, "test.csv")
    train_file = os.path.join(instance_dir, "train.csv")
    
    if not os.path.exists(test_file):
        return None, f"Test file not found: {test_file}", ""
    if not os.path.exists(train_file):
        return None, f"Train file not found: {train_file}", ""
    
    cmd = [
        PYTHON_EXECUTABLE, script_path,
        "--demand-file", test_file,
        "--promised-lead-time", str(promised_lead_time),
        "--real-instance-train", train_file,
    ]
    if max_periods is not None:
        cmd.extend(["--max-periods", str(max_periods)])
    
    try:
        result = subprocess.run(
            cmd,
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=2000,  # 33 minute timeout (2000 seconds)
            stdin=subprocess.DEVNULL,  # Prevent waiting for stdin input
        )
        
        output = result.stdout + result.stderr
        
        # Save output to log file
        log_filename = f"{script_name}_{run_num}.txt"
        log_path = os.path.join(instance_dir, log_filename)
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(output)
        except Exception as e:
            # If we can't write the log, continue anyway but note it
            output += f"\n[Warning: Could not save log to {log_path}: {str(e)}]"
        
        reward = extract_reward_from_output(output)
        
        if reward is None:
            return None, f"Could not extract reward from output. Exit code: {result.returncode}", output
        
        if result.returncode != 0:
            return None, f"Script failed with exit code {result.returncode}", output
        
        return reward, None, output
        
    except subprocess.TimeoutExpired as e:
        timeout_minutes = 2000 / 60
        return None, f"Script timed out after {timeout_minutes:.1f} minutes ({2000} seconds)", ""
    except Exception as e:
        return None, f"Error running script: {str(e)}", ""


def run_single_task(task_info):
    """Wrapper function for parallel execution."""
    script_path, run_num, script_name, promised_lead_time, instance_dir, max_periods, base_dir = task_info
    return (script_name, run_num), run_script(
        script_path, run_num, script_name, promised_lead_time, instance_dir, max_periods, base_dir
    )


def benchmark_all(promised_lead_time: int, instance_dir: str, max_periods: int = None, max_workers: int = None):
    """Run all benchmarks and collect results."""
    
    # Validate instance directory
    instance_dir = os.path.abspath(instance_dir)
    if not os.path.exists(instance_dir):
        print(f"Error: Directory not found: {instance_dir}")
        return
    
    test_file = os.path.join(instance_dir, "test.csv")
    train_file = os.path.join(instance_dir, "train.csv")
    
    if not os.path.exists(test_file):
        print(f"Error: test.csv not found in {instance_dir}")
        return
    if not os.path.exists(train_file):
        print(f"Error: train.csv not found in {instance_dir}")
        return
    
    instance_name = os.path.basename(instance_dir)
    
    # Determine which scripts use LLM (for parallel processing)
    llm_scripts = {"llm", "llm_to_or", "or_to_llm"}
    or_scripts = {"or"}
    
    # Set default max_workers for LLM scripts
    if max_workers is None:
        max_workers = min(30, (os.cpu_count() or 4) * 3)
    
    print("=" * 80)
    print("BENCHMARK CONFIGURATION")
    print("=" * 80)
    print(f"Instance directory: {instance_dir}")
    print(f"Instance name: {instance_name}")
    print(f"Promised lead time: {promised_lead_time}")
    print(f"Strategies: {', '.join(SCRIPTS.keys())}")
    print(f"Runs per strategy: {NUM_RUNS}")
    print(f"Total runs: {len(SCRIPTS) * NUM_RUNS}")
    if max_periods:
        print(f"Max periods: {max_periods}")
    else:
        print(f"Max periods: All (no limit)")
    print(f"Parallel workers: {max_workers}")
    print("=" * 80)
    
    # Results structure: results[script_name] = [reward1, reward2, ...]
    results = defaultdict(list)
    errors = defaultdict(list)
    
    total_runs = len(SCRIPTS) * NUM_RUNS
    completed_runs = 0
    
    # Prepare all LLM tasks upfront
    all_llm_tasks = []
    for script_name, script_path in SCRIPTS.items():
        if script_name in llm_scripts:
            script_full_path = BASE_DIR / script_path
            if script_full_path.exists():
                for run_num in range(1, NUM_RUNS + 1):
                    all_llm_tasks.append((
                        str(script_full_path), run_num, script_name,
                        promised_lead_time, instance_dir, max_periods, str(BASE_DIR)
                    ))
    
    # Run all LLM scripts in parallel
    if all_llm_tasks:
        print(f"\nRunning {len(all_llm_tasks)} LLM tasks in parallel ({max_workers} workers)...")
        print("This includes: llm, llm_to_or, or_to_llm\n")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(run_single_task, task): task for task in all_llm_tasks}
            
            for future in as_completed(future_to_task):
                completed_runs += 1
                task = future_to_task[future]
                script_name = task[2]
                run_num = task[1]
                
                try:
                    _, (reward, error, output) = future.result()
                    
                    if error:
                        print(f"[{script_name}] Run {run_num}/{NUM_RUNS} ({completed_runs}/{len(all_llm_tasks)}): ERROR: {error}")
                        errors[script_name].append(error)
                    else:
                        print(f"[{script_name}] Run {run_num}/{NUM_RUNS} ({completed_runs}/{len(all_llm_tasks)}): Reward: ${reward:.2f} (log saved)")
                        results[script_name].append(reward)
                except Exception as e:
                    print(f"[{script_name}] Run {run_num}/{NUM_RUNS} ({completed_runs}/{len(all_llm_tasks)}): EXCEPTION: {str(e)}")
                    errors[script_name].append(f"Exception: {str(e)}")
    
    # Run OR script sequentially (fast, no need for parallel)
    print(f"\n{'='*80}")
    print("Running OR script sequentially (fast execution)...")
    print(f"{'='*80}\n")
    
    script_name = "or"
    script_path = SCRIPTS[script_name]
    script_full_path = BASE_DIR / script_path
    
    if not script_full_path.exists():
        print(f"[{script_name}] ERROR: Script not found: {script_full_path}")
    else:
        print(f"[{script_name}] Running {NUM_RUNS} runs...")
        
        for run_num in range(1, NUM_RUNS + 1):
            completed_runs += 1
            print(f"  Run {run_num}/{NUM_RUNS} ({completed_runs}/{total_runs})...", end=" ", flush=True)
            
            reward, error, output = run_script(
                str(script_full_path), run_num, script_name,
                promised_lead_time, instance_dir, max_periods
            )
            
            if error:
                print(f"ERROR: {error}")
                errors[script_name].append(error)
            else:
                print(f"Reward: ${reward:.2f} (log saved)")
                results[script_name].append(reward)
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nInstance: {instance_name}")
    print(f"Promised Lead Time: {promised_lead_time}")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Avg Reward':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15} {'Runs':<10}")
    print("-" * 80)
    
    overall_stats = {}
    for script_name in SCRIPTS.keys():
        rewards = results[script_name]
        if rewards:
            overall_stats[script_name] = {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'count': len(rewards),
            }
            print(f"{script_name:<20} ${overall_stats[script_name]['mean']:<14.2f} "
                  f"${overall_stats[script_name]['std']:<14.2f} "
                  f"${overall_stats[script_name]['min']:<14.2f} "
                  f"${overall_stats[script_name]['max']:<14.2f} "
                  f"{overall_stats[script_name]['count']:<10}")
        else:
            print(f"{script_name:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'0':<10}")
    
    # Show errors if any
    print("\n" + "-" * 80)
    has_errors = False
    for script_name in SCRIPTS.keys():
        if errors[script_name]:
            has_errors = True
            print(f"\nErrors for {script_name}:")
            for error in errors[script_name]:
                print(f"  - {error}")
    
    if not has_errors:
        print("\nNo errors encountered.")
    
    # Save detailed results to JSON
    output_file = os.path.join(instance_dir, "benchmark_results.json")
    detailed_results = {
        'instance_dir': instance_dir,
        'instance_name': instance_name,
        'promised_lead_time': promised_lead_time,
        'num_runs': NUM_RUNS,
        'max_periods': max_periods,
        'results': {
            script_name: {
                'rewards': [float(r) for r in rewards],
                'mean': float(np.mean(rewards)) if rewards else None,
                'std': float(np.std(rewards)) if rewards else None,
                'min': float(np.min(rewards)) if rewards else None,
                'max': float(np.max(rewards)) if rewards else None,
                'count': len(rewards),
            }
            for script_name, rewards in {
                script_name: results[script_name]
                for script_name in SCRIPTS.keys()
            }.items()
        },
        'errors': {
            script_name: errors[script_name]
            for script_name in SCRIPTS.keys()
            if errors[script_name]
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n\nDetailed results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Benchmark all strategies (or, llm, llm_to_or, or_to_llm)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python benchmark_all_strategies.py --promised-lead-time 1 --directory D:\\TextArena\\examples\\initial_synthetic_demand_files\\case1_iid_normal
        """
    )
    parser.add_argument('--promised-lead-time', type=int, required=True,
                       help='Promised lead time in periods (required)')
    parser.add_argument('--directory', type=str, required=True,
                       help='Path to instance directory containing test.csv and train.csv (required)')
    parser.add_argument('--max-periods', type=int, default=None,
                       help='Maximum number of periods to run per test. Default: None (runs all periods)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers for LLM scripts. Default: min(20, CPU_count * 3)')
    args = parser.parse_args()
    
    benchmark_all(
        promised_lead_time=args.promised_lead_time,
        instance_dir=args.directory,
        max_periods=args.max_periods,
        max_workers=args.max_workers
    )
