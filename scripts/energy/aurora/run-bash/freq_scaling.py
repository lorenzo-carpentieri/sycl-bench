import argparse
import subprocess
import time
import getpass
# 1. run black scholes and matrix mul with geopm frequency scaling with all the available frequency of the hw
# 2. read and parse the output generated for each frequency
# 3. generate plto of frequency scaling y-axis normalized energy and x-axis speedup

# Freq. info
SAMPLING_FREQ_FACTOR=1
FREQ_STEP=50 # for intel gpu the step is 50
MAX_FREQ_AVAIL=1600
MIN_FREQ_AVAIL=200

bench_sizes = {
    'black_scholes': 524288,
    'matrix_mul': 5000
}

bench_num_iters = {
    'black_scholes': 500000 ,
    'matrix_mul': 5,
}


def wait_for_jobs():
    user = getpass.getuser()
    while True:
        result = subprocess.run(
            ['qstat', '-u', 'lcarpent'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        # Count lines except header
        lines = result.stdout.strip().split('\n')
        # If only header is present, no jobs are running
        if len(lines) <= 1:
            break
        print("Waiting for jobs to finish...")
        time.sleep(30)  # Wait 30 seconds before checking again


def main():
    parser = argparse.ArgumentParser(description="Frequency scaling analysis for Black-Scholes and Matrix Multiplication benchmarks.")
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to store output results')
    parser.add_argument('--app-path', type=str, required=True, help='Directory to store output results')
    parser.add_argument('--benchmarks', nargs='+', default=['black_scholes', 'matrix_mul'], help='Benchmarks to run')
    parser.add_argument('--pbs-path', type=str, required=True, help='path to pbs script for submitting the job on the aurora queue')
    
    args = parser.parse_args()

    print(f"Output directory: {args.output_dir}")
    print(f"App directory: {args.app_path}")
    print(f"Benchmarks: {args.benchmarks}")
    benchmarks = args.benchmarks
    output_dir = args.output_dir
    app_path = args.app_path
    pbs_path = args.pbs_path

    num_runs = 10  # Number of runs for each benchmark at each frequency
    for bench in benchmarks:
        print(f"Running benchmark: {bench}")
        full_app_path = f"{app_path}/{bench}"
        for freq in range(MIN_FREQ_AVAIL, MAX_FREQ_AVAIL + FREQ_STEP, FREQ_STEP*SAMPLING_FREQ_FACTOR):
            print(f"  Running {bench} with {freq} MHz")
            # Construct the command to submit the job via qsub
            cmd = [
                "qsub",
                f"-v path_app={full_app_path},core_freq={freq},size={bench_sizes[bench]},num_iters={bench_num_iters[bench]},num_runs={num_runs},path_out={output_dir}",
                pbs_path,
            ]

            print(f"    Submitting job: {' '.join(cmd)}")
            subprocess.run(cmd)
            wait_for_jobs()
           

        if freq < MAX_FREQ_AVAIL:
            print(f"  Running {bench} with {MAX_FREQ_AVAIL} MHz")
                # Construct the command to submit the job via qsub
            cmd = [
                "qsub",
                f"-v path_app={full_app_path},core_freq={MAX_FREQ_AVAIL},size={bench_sizes[bench]},num_iters={bench_num_iters[bench]},num_runs={num_runs},path_out={output_dir}",
                pbs_path,
            ]

            print(f"    Submitting job: {' '.join(cmd)}")
            subprocess.run(cmd)
            wait_for_jobs()


if __name__ == "__main__":
    main()
