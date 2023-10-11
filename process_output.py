import csv
import json
import os

slurm_log_path = "{0}ocean{0}projects{0}hum160002p{0}shared{0}books{0}code{0}test_workflow{0}logs{0}".format(os.sep)
errored_logs_file = "process_output.json"

slurm_logs = [
    "slurm-19718815.out",
    "slurm-19718820.out",
    "slurm-19718821.out",
    "slurm-19718822.out",
    "slurm-19718823.out",
    "slurm-19718824.out",
    "slurm-19718825.out",
    "slurm-19718826.out",
    "slurm-19718828.out",
    "slurm-19718829.out",
    "slurm-19718830.out",
    "slurm-19718831.out",
    "slurm-19718832.out",
    "slurm-19718833.out",
    "slurm-19718834.out",
    "slurm-19718835.out",
    "slurm-19718836.out",
    "slurm-19718837.out",
    "slurm-19718838.out",
    "slurm-19718839.out",
    "slurm-19718840.out",
    "slurm-19718841.out",
    "slurm-19718842.out",
    "slurm-19718843.out",
    "slurm-19718844.out",
    "slurm-19718845.out",
    "slurm-19718846.out",
    "slurm-19718847.out",
    "slurm-19718848.out",
    "slurm-19718849.out",
    "slurm-19718850.out",
    "slurm-19718851.out",
    "slurm-19718852.out",
    "slurm-19718853.out",
    "slurm-19718854.out",
    "slurm-19718855.out",
    "slurm-19718856.out",
    "slurm-19718857.out",
    "slurm-19718858.out",
    "slurm-19718859.out",
    "slurm-19718860.out",
    "slurm-19718861.out",
    "slurm-19718862.out",
    "slurm-19718863.out",
    "slurm-19718864.out",
    "slurm-19718865.out",
    "slurm-19718866.out",
    "slurm-19718867.out",
    "slurm-19718868.out",
    "slurm-19718869.out",
    "slurm-19718870.out",
    "slurm-19718871.out",
    "slurm-19718872.out",
    "slurm-19718873.out",
    "slurm-19718874.out",
    "slurm-19718875.out",
    "slurm-19718876.out",
    "slurm-19718878.out",
    "slurm-19718879.out",
    "slurm-19718880.out",
    "slurm-19718881.out",
    "slurm-19718882.out",
    "slurm-19718883.out",
    "slurm-19718884.out",
    "slurm-19718885.out",
    "slurm-19718886.out",
    "slurm-19718887.out",
    "slurm-19718888.out",
    "slurm-19718889.out",
    "slurm-19718890.out",
    "slurm-19718891.out",
    "slurm-19718892.out",
    "slurm-19718893.out",
    "slurm-19718894.out",
    "slurm-19718895.out",
    "slurm-19718899.out",
    "slurm-19718900.out",
    "slurm-19718902.out",
    "slurm-19718903.out",
    "slurm-19718904.out",
    "slurm-19718905.out",
    "slurm-19718906.out",
    "slurm-19718907.out",
    "slurm-19718908.out",
    "slurm-19718909.out",
    "slurm-19718910.out",
    "slurm-19718911.out",
    "slurm-19718912.out",
    "slurm-19718913.out",
    "slurm-19718914.out",
    "slurm-19718915.out",
    "slurm-19718916.out",
    "slurm-19718917.out",
    "slurm-19718918.out",
    "slurm-19718919.out",
    "slurm-19718920.out",
    "slurm-19718921.out",
    "slurm-19718922.out"
]
run_master_log = "slurm-19718815.out"

error_keywords = [
    "out-of-memory",
    "index -1 is out of bounds",
    "list index out of range",
    "PIL.UnidentifiedImageError",
    "cv2.error",
    "ValueError: v cannot be empty",
],

success_line_beginnings = [

    "Creating",
    "Submitted",
    "no change",
    "No action",
    "Args for",
    "Pages:",
    "Cropping",
    "Book dir:",
    "Book name:",
    "/ocean"
]

def find_errored_logs():

    # Gather files with errors and their error lines
    files_with_errors = {}
    for log_filename in slurm_logs:

        with open(slurm_log_path + log_filename, "r") as log_file:

            log_lines = log_file.readlines()
            error_lines = []

            for line in log_lines:

                success_line = False

                for beginning in success_line_beginnings:

                    if line.strip().startswith(beginning) or 0 == len(line.strip()):
                        success_line = True
                        break

                if not success_line:
                    error_lines.append(line)

            if len(error_lines):
                files_with_errors[log_filename] = error_lines

    with open(os.getcwd() + os.sep + "process_output.json", "w") as output_file:
        json.dump(files_with_errors, output_file, indent=4)

def analyze_successful_logs():

    # 1. Produce a list of slurm log files representing successful runs
    with open(os.getcwd() + os.sep + errored_logs_file, "r") as input_file:
        errors_json = json.load(input_file)
    successful_logs = list(set(slurm_log_list) - set(errors_json.keys()) - set([run_master_log]))

    # 2. Produce a csv detailing the successful logs

    # A. Gather information on each successful run
    slog_json = {}
    for filename in successful_logs:
        
        slog_json[filename] = {}
        
        with open("{0}{1}logs{1}{2}".format(os.getcwd(), os.sep, filename), "r") as log_file:

            print(filename)

            log_lines = log_file.readlines()
            for line in log_lines:
                if line.startswith("Args for"):
                    slog_json[filename]["threshold_by_inside"] = "threshold_by_inside=True" in line
                elif line.startswith("Book dir:"):
                    slog_json[filename]["book_dir"] = line[line.find(": ") + 1:].strip()
                elif line.startswith("Book name:"):
                    slog_json[filename]["book_name"] = line[line.find(": ") + 1:].strip()
        
    # 3. Output information on successful runs in one csv
    with open(os.getcwd() + os.sep + "successful_runs.csv", "w") as output_file:

        csv_writer = csv.writer(output_file)

        csv_writer.writerow(["book_name", "crop_type", "book_dir", "log_filepath"])

        for filename in successful_logs:
            csv_writer.writerow([
                slog_json[filename]["book_name"],
                "threshold_by_inside" if slog_json[filename]["threshold_by_inside"] else "non_threshold_by_inside",
                slog_json[filename]["book_dir"],
                "{0}{1}logs{1}{2}".format(os.getcwd(), os.sep, filename)
            ])


# find_errored_logs()
analyze_successful_logs()