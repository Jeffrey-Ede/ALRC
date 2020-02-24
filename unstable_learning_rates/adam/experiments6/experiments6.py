import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LOG_DIR = "//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/modified_losses/experiments6/"
CIFAR10_DIR = "//ads.warwick.ac.uk/shared/HCSS6/Shared305/Microscopy/Jeffrey-Ede/models/modified_losses/cifar10-6/"
NUM_REPEATS = 1
MOMENT_DECAY = 0.999
BASE_LEARNING_RATE = None

def run_experiments(initial_lr, num_stddev, repeat_num):
        
    #Training program
    command = f"py {CIFAR10_DIR}cifar10_train.py"

    #Task hyperparameters
    command += f" --batch_size 1"
    command += f" --num_stddev {num_stddev}"
    command += f" --initial_lr {initial_lr}"

    #Loss log
    log_dir = f"{LOG_DIR}batch_size/{initial_lr}/num_stddev/{num_stddev}/repeat/{repeat_num}"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log.txt"
    command += f" --log_file {log_file}"

    os.system(command)

for repeat_num in range(NUM_REPEATS):
    for num_stddev in [0, 2]:
        for initial_lr in [0.0001, 0.002]:
            run_experiments(initial_lr=initial_lr, 
                            num_stddev=num_stddev,
                            repeat_num=repeat_num)


