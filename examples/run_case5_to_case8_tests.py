import os
import sys
import subprocess
from pathlib import Path
import argparse


def run_command_and_save_output(command_args, output_file_path: Path) -> int:
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with output_file_path.open('w', encoding='utf-8') as output_file:
        completed = subprocess.run(
            command_args,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
            text=True,
            shell=False,
            check=False,
        )
        returncode = completed.returncode
        if returncode != 0:
            # Append failure marker to the same file for quick triage
            output_file.write("\n\n[RUNNER NOTE] Command exited with non-zero status: " + str(returncode) + "\n")
            output_file.write("[RUNNER NOTE] Command: " + ' '.join(command_args) + "\n")
    return returncode


def main() -> None:
    parser = argparse.ArgumentParser(description='Run cases 5-8 with uv run and save outputs')
    parser.add_argument('--only', nargs='+', type=int, help='Specify case numbers to run (subset of 5 6 7 8)')
    args = parser.parse_args()
    # Ensure API key present for LLM-based runs
    if not os.getenv('OPENAI_API_KEY'):
        print('Error: OPENAI_API_KEY not set. Set it before running to enable LLM-based tests.')
        print('Example (PowerShell): $env:OPENAI_API_KEY = "sk-your-key"')
        sys.exit(1)

    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / 'examples'

    # Demand CSVs for case5-8
    case_to_csv = {
        5: examples_dir / 'demand_case5_L_const4.csv',
        6: examples_dir / 'demand_case6_L_normal_4_2.csv',
        7: examples_dir / 'demand_case7_L_shift_cp15.csv',
        8: examples_dir / 'demand_case8_L_inf_p01.csv',
    }

    # Validate CSVs exist
    missing = [str(p) for p in case_to_csv.values() if not p.exists()]
    if missing:
        print('Missing CSV files:')
        for m in missing:
            print(f'  - {m}')
        sys.exit(1)

    # Scripts
    uv = 'uv'  # rely on uv being on PATH, consistent with your manual runs
    llm_script = str((examples_dir / 'llm_csv_demo.py').resolve())
    or_script = str((examples_dir / 'or_csv_demo.py').resolve())
    or_to_llm_script = str((examples_dir / 'or_to_llm_csv_demo.py').resolve())
    llm_to_or_script = str((examples_dir / 'llm_to_or_csv_demo.py').resolve())

    def llm_cmd(demand_csv: Path):
        return [
            uv, 'run', llm_script,
            '--demand-file', str(demand_csv.resolve()),
            '--promised-lead-time', '4',
        ]

    def or_cmd(demand_csv: Path):
        return [
            uv, 'run', or_script,
            '--demand-file', str(demand_csv.resolve()),
            '--promised-lead-time', '4',
        ]

    def or_to_llm_cmd(demand_csv: Path):
        return [
            uv, 'run', or_to_llm_script,
            '--demand-file', str(demand_csv.resolve()),
            '--promised-lead-time', '4',
        ]

    def llm_to_or_cmd(demand_csv: Path):
        return [
            uv, 'run', llm_to_or_script,
            '--demand-file', str(demand_csv.resolve()),
            '--promised-lead-time', '4',
        ]

    # Run cases 5 to 8
    run_cases = args.only if args.only else [5, 6, 7, 8]
    for case_num, csv_path in case_to_csv.items():
        if case_num not in run_cases:
            continue
        out_dir = examples_dir / f'case{case_num}_tests'
        print(f'Running case{case_num} tests...')

        # LLM (2 runs)
        run_command_and_save_output(llm_cmd(csv_path), out_dir / 'llm_1.txt')
        run_command_and_save_output(llm_cmd(csv_path), out_dir / 'llm_2.txt')

        # OR (1 run)
        run_command_and_save_output(or_cmd(csv_path), out_dir / 'or.txt')

        # OR -> LLM (2 runs)
        run_command_and_save_output(or_to_llm_cmd(csv_path), out_dir / 'or_to_llm_1.txt')
        run_command_and_save_output(or_to_llm_cmd(csv_path), out_dir / 'or_to_llm_2.txt')

        # LLM -> OR (2 runs)
        run_command_and_save_output(llm_to_or_cmd(csv_path), out_dir / 'llm_to_or_1.txt')
        run_command_and_save_output(llm_to_or_cmd(csv_path), out_dir / 'llm_to_or_2.txt')

    print('Done. Outputs saved to:')
    for case_num in run_cases:
        print(f'  {examples_dir / f"case{case_num}_tests"}')


if __name__ == '__main__':
    main()


