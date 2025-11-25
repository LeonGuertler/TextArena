"""
Benchmark script to compare all strategies across H&M instances.

Runs each strategy 5 times per instance and calculates:
- Average reward
- Standard deviation
- Min/Max rewards
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

# Scripts to benchmark
SCRIPTS = {
    "llm": "examples/llm_csv_demo.py",
    "or": "examples/or_csv_demo.py",
    "or_to_llm": "examples/or_to_llm_csv_demo.py",
}

# Number of runs per script-instance combination
NUM_RUNS = 5

# Base directory
BASE_DIR = Path(__file__).parent.parent
H_M_INSTANCES_DIR = BASE_DIR / "examples" / "H&M_instances"


def extract_reward_from_output(output: str) -> float:
    """Extract total reward from script output."""
    # Pattern to match: >>> Total Reward: $1234.56 <<<
    # or variations like "Total Reward (OR Baseline): $1234.56"
    patterns = [
        r">>>\s*Total Reward[^:]*:\s*\$([\d,]+\.?\d*)\s*<<<",  # With >>> ... <<<
        r"Total Reward[^:]*:\s*\$([\d,]+\.?\d*)",  # Without >>> ... <<<
        r"VM Final Reward:\s*([\d,]+\.?\d*)",  # Fallback to VM Final Reward
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            try:
                # Remove commas from number string
                reward_str = match.group(1).replace(',', '')
                return float(reward_str)
            except ValueError:
                continue
    
    # If no match found, try to find any number after "Total Reward"
    lines = output.split('\n')
    for line in lines:
        if 'Total Reward' in line or 'total_reward' in line.lower():
            # Try to extract number (handle comma-separated numbers)
            numbers = re.findall(r'[\d,]+\.?\d*', line)
            if numbers:
                try:
                    # Take the last number and remove commas
                    reward_str = numbers[-1].replace(',', '')
                    return float(reward_str)
                except ValueError:
                    continue
    
    return None


def run_script(script_path: str, instance_id: str, run_num: int, script_name: str,
               promised_lead_time: int = 1, max_periods: int = None, 
               base_dir: str = None, h_m_instances_dir: str = None) -> Tuple[float, str, str]:
    """
    Run a script for a given instance and return the reward and output.
    
    Returns:
        (reward, error_message, output)
    """
    # Use provided paths or fall back to module-level constants
    if base_dir is None:
        base_dir = str(BASE_DIR)
    if h_m_instances_dir is None:
        h_m_instances_dir = str(H_M_INSTANCES_DIR)
    
    test_file = os.path.join(h_m_instances_dir, instance_id, "test.csv")
    train_file = os.path.join(h_m_instances_dir, instance_id, "train.csv")
    instance_dir = os.path.join(h_m_instances_dir, instance_id)
    
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
            timeout=600,  # 10 minute timeout
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
        
    except subprocess.TimeoutExpired:
        return None, "Script timed out after 10 minutes", ""
    except Exception as e:
        return None, f"Error running script: {str(e)}", ""


def get_all_instances() -> List[str]:
    """Get all instance IDs from H&M_instances directory."""
    # Only test these specific instances
    target_instances = ["706016001", "568601006", "599580017"]
    
    instances = []
    if not H_M_INSTANCES_DIR.exists():
        print(f"Error: H&M_instances directory not found at {H_M_INSTANCES_DIR}")
        return instances
    
    for instance_id in target_instances:
        instance_path = H_M_INSTANCES_DIR / instance_id
        if instance_path.exists() and instance_path.is_dir() and (instance_path / "test.csv").exists():
            instances.append(instance_id)
        else:
            print(f"Warning: Instance {instance_id} not found or missing test.csv")
    
    return sorted(instances)


def run_single_task(task_info):
    """Wrapper function for parallel execution."""
    script_path, instance_id, run_num, script_name, promised_lead_time, max_periods, base_dir, h_m_instances_dir = task_info
    return (script_path, instance_id, run_num), run_script(
        script_path, instance_id, run_num, script_name, promised_lead_time, max_periods, base_dir, h_m_instances_dir
    )


def benchmark_all(max_periods=None, max_workers=None):
    """Run all benchmarks and collect results."""
    instances = get_all_instances()
    
    if not instances:
        print("No instances found!")
        return
    
    # Determine which scripts use LLM (for parallel processing)
    llm_scripts = {"llm", "or_to_llm"}
    or_scripts = {"or"}
    
    # Set default max_workers for LLM scripts
    if max_workers is None:
        max_workers = min(15, (os.cpu_count() or 4) * 3)  # Allow more workers for parallel LLM scripts
    
    print(f"Found {len(instances)} instances: {instances}")
    print(f"Running {NUM_RUNS} runs per script-instance combination...")
    print(f"Total runs: {len(instances) * len(SCRIPTS) * NUM_RUNS}")
    if max_periods:
        print(f"Limiting to {max_periods} periods per run")
    else:
        print(f"Running all periods from test files (no limit)")
    print(f"Parallel processing: {max_workers} workers for LLM scripts (llm, or_to_llm)")
    print("=" * 80)
    
    # Results structure: results[script_name][instance_id] = [reward1, reward2, ...]
    results = defaultdict(lambda: defaultdict(list))
    errors = defaultdict(lambda: defaultdict(list))
    
    total_runs = len(instances) * len(SCRIPTS) * NUM_RUNS
    completed_runs = 0
    
    # Prepare all LLM tasks upfront (all scripts, all instances, all runs)
    all_llm_tasks = []
    for instance_id in instances:
        for script_name, script_path in SCRIPTS.items():
            if script_name in llm_scripts:
                script_full_path = BASE_DIR / script_path
                if script_full_path.exists():
                    for run_num in range(1, NUM_RUNS + 1):
                        all_llm_tasks.append((
                            str(script_full_path), instance_id, run_num, script_name, 
                            1, max_periods, str(BASE_DIR), str(H_M_INSTANCES_DIR)
                        ))
    
    # Run all LLM scripts in parallel
    if all_llm_tasks:
        print(f"\nRunning {len(all_llm_tasks)} LLM tasks in parallel ({max_workers} workers)...")
        print("This includes: llm and or_to_llm across all instances\n")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(run_single_task, task): task for task in all_llm_tasks}
            
            # Track progress by script and instance
            progress = defaultdict(lambda: defaultdict(int))
            
            for future in as_completed(future_to_task):
                completed_runs += 1
                task = future_to_task[future]
                script_path, instance_id, run_num, script_name = task[0], task[1], task[2], task[3]
                
                try:
                    _, (reward, error, output) = future.result()
                    progress[script_name][instance_id] += 1
                    
                    if error:
                        print(f"[{script_name}/{instance_id}] Run {run_num}/{NUM_RUNS} ({completed_runs}/{len(all_llm_tasks)}): ERROR: {error}")
                        if instance_id not in errors[script_name]:
                            errors[script_name][instance_id] = []
                        errors[script_name][instance_id].append(error)
                    else:
                        print(f"[{script_name}/{instance_id}] Run {run_num}/{NUM_RUNS} ({completed_runs}/{len(all_llm_tasks)}): Reward: ${reward:.2f} (log saved)")
                        if instance_id not in results[script_name]:
                            results[script_name][instance_id] = []
                        results[script_name][instance_id].append(reward)
                except Exception as e:
                    print(f"[{script_name}/{instance_id}] Run {run_num}/{NUM_RUNS} ({completed_runs}/{len(all_llm_tasks)}): EXCEPTION: {str(e)}")
                    if instance_id not in errors[script_name]:
                        errors[script_name][instance_id] = []
                    errors[script_name][instance_id].append(f"Exception: {str(e)}")
    
    # Run OR script sequentially (fast, no need for parallel)
    print(f"\n{'='*80}")
    print("Running OR script sequentially (fast execution)...")
    print(f"{'='*80}\n")
    
    for instance_id in instances:
        script_name = "or"
        script_path = SCRIPTS[script_name]
        script_full_path = BASE_DIR / script_path
        
        if not script_full_path.exists():
            print(f"[{script_name}/{instance_id}] ERROR: Script not found: {script_full_path}")
            continue
        
        print(f"[{script_name}/{instance_id}] Running {NUM_RUNS} runs...")
        instance_rewards = []
        instance_errors = []
        
        for run_num in range(1, NUM_RUNS + 1):
            completed_runs += 1
            print(f"  Run {run_num}/{NUM_RUNS} ({completed_runs}/{total_runs})...", end=" ", flush=True)
            
            reward, error, output = run_script(str(script_full_path), instance_id, run_num, script_name, 1, max_periods)
            
            if error:
                print(f"ERROR: {error}")
                instance_errors.append(error)
            else:
                print(f"Reward: ${reward:.2f} (log saved)")
                instance_rewards.append(reward)
        
        results[script_name][instance_id] = instance_rewards
        if instance_errors:
            errors[script_name][instance_id] = instance_errors
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Overall statistics per script
    print("\nOverall Statistics (across all instances):")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Avg Reward':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15} {'Runs':<10}")
    print("-" * 80)
    
    overall_stats = {}
    for script_name in SCRIPTS.keys():
        all_rewards = []
        for instance_id in instances:
            all_rewards.extend(results[script_name][instance_id])
        
        if all_rewards:
            overall_stats[script_name] = {
                'mean': np.mean(all_rewards),
                'std': np.std(all_rewards),
                'min': np.min(all_rewards),
                'max': np.max(all_rewards),
                'count': len(all_rewards),
            }
            print(f"{script_name:<20} ${overall_stats[script_name]['mean']:<14.2f} "
                  f"${overall_stats[script_name]['std']:<14.2f} "
                  f"${overall_stats[script_name]['min']:<14.2f} "
                  f"${overall_stats[script_name]['max']:<14.2f} "
                  f"{overall_stats[script_name]['count']:<10}")
    
    # Per-instance statistics
    print("\n\nPer-Instance Statistics:")
    print("-" * 80)
    
    for instance_id in instances:
        print(f"\nInstance: {instance_id}")
        print(f"{'Strategy':<20} {'Avg Reward':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15} {'Runs':<10}")
        print("-" * 80)
        
        for script_name in SCRIPTS.keys():
            rewards = results[script_name][instance_id]
            if rewards:
                mean = np.mean(rewards)
                std = np.std(rewards)
                min_reward = np.min(rewards)
                max_reward = np.max(rewards)
                print(f"{script_name:<20} ${mean:<14.2f} ${std:<14.2f} "
                      f"${min_reward:<14.2f} ${max_reward:<14.2f} {len(rewards):<10}")
            else:
                print(f"{script_name:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'0':<10}")
        
        # Show errors if any
        for script_name in SCRIPTS.keys():
            if errors[script_name][instance_id]:
                print(f"\n  Errors for {script_name}:")
                for error in errors[script_name][instance_id]:
                    print(f"    - {error}")
    
    # Save detailed results to JSON
    output_file = BASE_DIR / "examples" / "benchmark_results.json"
    detailed_results = {
        'overall_stats': {k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv 
                              for kk, vv in v.items()} 
                          for k, v in overall_stats.items()},
        'per_instance': {
            instance_id: {
                script_name: {
                    'rewards': [float(r) for r in rewards],
                    'mean': float(np.mean(rewards)) if rewards else None,
                    'std': float(np.std(rewards)) if rewards else None,
                    'min': float(np.min(rewards)) if rewards else None,
                    'max': float(np.max(rewards)) if rewards else None,
                    'count': len(rewards),
                }
                for script_name, rewards in {
                    script_name: results[script_name][instance_id]
                    for script_name in SCRIPTS.keys()
                }.items()
            }
            for instance_id in instances
        },
        'errors': {
            script_name: {
                instance_id: errors[script_name][instance_id]
                for instance_id in instances
                if errors[script_name][instance_id]
            }
            for script_name in SCRIPTS.keys()
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n\nDetailed results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark all strategies across H&M instances')
    parser.add_argument('--max-periods', type=int, default=None,
                       help='Maximum number of periods to run per test. Default: None (runs all periods from test files)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers for LLM scripts. Default: min(15, CPU_count * 3)')
    args = parser.parse_args()
    
    benchmark_all(max_periods=args.max_periods, max_workers=args.max_workers)

