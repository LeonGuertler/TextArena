import os
import sys
import subprocess
from pathlib import Path


def run_command_and_save_output(command_args, output_file_path: Path) -> None:
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
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command_args)}\nSee: {output_file_path}")


def main() -> None:
    # Ensure API key present for LLM-based runs
    if not os.getenv('OPENAI_API_KEY'):
        print('Error: OPENAI_API_KEY not set. Set it before running to enable LLM-based tests.')
        print('Example (PowerShell): $env:OPENAI_API_KEY = "sk-your-key"')
        sys.exit(1)

    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / 'examples'

    # Demand CSVs for case3 and case4
    demand_case3 = examples_dir / 'demand_case3_increasing.csv'
    demand_case4 = examples_dir / 'demand_case4_normal_to_uniform_cp15.csv'

    if not demand_case3.exists():
        print(f'Missing CSV: {demand_case3}')
        sys.exit(1)
    if not demand_case4.exists():
        print(f'Missing CSV: {demand_case4}')
        sys.exit(1)

    # Output directories
    case3_out = examples_dir / 'case3_tests'
    case4_out = examples_dir / 'case4_tests'

    # Scripts
    py = sys.executable or 'python'
    llm_script = str(examples_dir / 'llm_csv_demo.py')
    or_script = str(examples_dir / 'or_csv_demo.py')
    or_to_llm_script = str(examples_dir / 'or_to_llm_csv_demo.py')
    llm_to_or_script = str(examples_dir / 'llm_to_or_csv_demo.py')

    # Common args (use defaults; promised lead time left as script default)
    def llm_cmd(demand_csv: Path):
        return [py, '-u', llm_script, '--demand-file', str(demand_csv)]

    def or_cmd(demand_csv: Path):
        return [py, '-u', or_script, '--demand-file', str(demand_csv)]

    def or_to_llm_cmd(demand_csv: Path):
        return [py, '-u', or_to_llm_script, '--demand-file', str(demand_csv)]

    def llm_to_or_cmd(demand_csv: Path):
        return [py, '-u', llm_to_or_script, '--demand-file', str(demand_csv)]

    # Case 3
    print('Running case3 tests...')
    run_command_and_save_output(llm_cmd(demand_case3), case3_out / 'llm_1.txt')
    run_command_and_save_output(llm_cmd(demand_case3), case3_out / 'llm_2.txt')
    run_command_and_save_output(or_cmd(demand_case3), case3_out / 'or.txt')
    run_command_and_save_output(or_to_llm_cmd(demand_case3), case3_out / 'or_to_llm_1.txt')
    run_command_and_save_output(or_to_llm_cmd(demand_case3), case3_out / 'or_to_llm_2.txt')
    run_command_and_save_output(llm_to_or_cmd(demand_case3), case3_out / 'llm_to_or_1.txt')
    run_command_and_save_output(llm_to_or_cmd(demand_case3), case3_out / 'llm_to_or_2.txt')

    # Case 4
    print('Running case4 tests...')
    run_command_and_save_output(llm_cmd(demand_case4), case4_out / 'llm_1.txt')
    run_command_and_save_output(llm_cmd(demand_case4), case4_out / 'llm_2.txt')
    run_command_and_save_output(or_cmd(demand_case4), case4_out / 'or.txt')
    run_command_and_save_output(or_to_llm_cmd(demand_case4), case4_out / 'or_to_llm_1.txt')
    run_command_and_save_output(or_to_llm_cmd(demand_case4), case4_out / 'or_to_llm_2.txt')
    run_command_and_save_output(llm_to_or_cmd(demand_case4), case4_out / 'llm_to_or_1.txt')
    run_command_and_save_output(llm_to_or_cmd(demand_case4), case4_out / 'llm_to_or_2.txt')

    print('Done. Outputs saved to:')
    print(f'  {case3_out}')
    print(f'  {case4_out}')


if __name__ == '__main__':
    main()


