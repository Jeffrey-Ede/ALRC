import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

LOG_DIR = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/modified_losses/experiments3/"
CIFAR10_DIR = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/modified_losses/cifar10-3/"
NUM_REPEATS = 10
MOMENT_DECAY = 0.999
BASE_LEARNING_RATE = None

def run_experiments(batch_size, num_stddev, repeat_num):
        
    #Training program
    command = f"py {CIFAR10_DIR}cifar10_train.py"

    #Task hyperparameters
    command += f" --batch_size {batch_size}"
    command += f" --num_stddev {num_stddev}"

    #Loss log
    log_dir = f"{LOG_DIR}batch_size/{batch_size}/num_stddev/{num_stddev}/repeat/{repeat_num}"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log.txt"
    command += f" --log_file {log_file}"

    os.system(command)

for repeat_num in range(NUM_REPEATS):
    for num_stddev in [2, 0, 3, 4]:
        for batch_size in [64, 1, 4, 16]:
                run_experiments(batch_size=batch_size, 
                                num_stddev=num_stddev,
                                repeat_num=repeat_num)


