import subprocess
import sys
import os

# Amount of times to run:
N = 300

# set the folder in which the cli.py is located
rerun_folder = os.path.abspath(os.path.dirname(__file__))

# the first one is the rerun.py script, the last is the output
used_arguments = sys.argv[1:]
output_location = os.path.abspath(sys.argv[-1])
for run_id in range(N):
    # in each run, the arguments are reused
    cmd = ["python", os.path.join(rerun_folder, "cli.py")]
    cmd.extend(used_arguments)
    subprocess.call(" ".join(cmd), shell=True)