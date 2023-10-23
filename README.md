# DeepSpeed Concurrent Jobs Launcher
## Overview
A Bash script starts multiple jobs simultaneously. Each job is an instance of AlexNet training over ImageNet using DeepSpeed. DeepSpeed is configured with CPU Offload.
## Components
### start-jobs-deepspeed.sh
A Bash script that starts multiple DeepSpeed jobs simultaneously. It is called with `./start-jobs-deepspeed.sh $number_jobs`. The stdout and stderr of each job is written to `job${number}.out`.

### model-deepspeed.py
A Python script that trains AlexNet over ImageNet. After every 2000 steps, it prints info, including RunningAvgSamplesPerSec and CurrSamplesPerSec. Some configurations are specified inside of this file as variables, including:

- momentum
- lr_decay
- lr_init
- num_epochs
- batch_size *(see note)*
- image_dim
- dataloader_num_workers
- device_ids *(see note)*
- num_classes
- train_img_dir

### ds_config.json
A minimal but working DeepSpeed configuration file for CPU Offload. For further explanation of what each configuration is doing, or more configurations that you would like to add, see [this page.](https://www.deepspeed.ai/docs/config-json/)
## Notes
### Batch Size
Batch size is configured as a variable in model-deepspeed.py **and** as a configuration in ds_config.json. Be sure to change it in both places.

### Device IDs
This experiment is currently set to use a single GPU. In order to change this, three different lines need to be changed. 
- In start-jobs-deepspeed.sh, `CUDA_VISIBLE_DEVICES=0`
- In model-deepspeed.py
	- `device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')`
	- `DEVICE_IDS = [0]  # GPUs to use`
## Steps
1. Install DeepSpeed.
	1. `pip install deepspeed`
	2. More info [here.](https://www.deepspeed.ai/getting-started/)
2. Obtain ImageNet dataset.
	1. More info [here.](https://image-net.org/download.php)
3. Clone this repository.
	1. `git clone https://github.com/builtbydc/deepspeed-concurrent-jobs`
4. Set configurations as described above.
	1. Set train_img_dir to ImageNet directory.
5. Run the experiment.
	1. `./start-jobs-deepspeed.sh $number_jobs`
