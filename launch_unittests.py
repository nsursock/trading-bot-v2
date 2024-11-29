import os
import subprocess
import argparse
import sys
from tabulate import tabulate  # Import tabulate

def run_tests_in_directory(directory_path, current_script):
    # Get a list of all .py files in the specified directory, excluding the current script
    py_files = [f for f in os.listdir(directory_path) if f.endswith('.py') and f != current_script]
    results = []

    for py_file in py_files:
        # Run each .py file as a separate process
        file_path = os.path.join(directory_path, py_file)
        try:
            subprocess.check_call(['python3', file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            results.append([py_file, "OK"])  # Test passed
        except subprocess.CalledProcessError:
            results.append([py_file, "FAIL"])  # Test failed

    # Print the results using tabulate
    print(tabulate(results, headers=["File", "Result"], tablefmt="pretty"))
    # print(tabulate(results, headers=["File", "Result"], tablefmt="grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Python unittests in a specified directory.')
    parser.add_argument('-d', '--directory', type=str, help='The directory containing the test files', default='.')
    args = parser.parse_args()

    # Get the name of the current script
    current_script = os.path.basename(sys.argv[0])

    run_tests_in_directory(args.directory, current_script)
