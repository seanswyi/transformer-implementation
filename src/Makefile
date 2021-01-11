debug:
	wandb off;
	python ./main.py --batch_size 2 --debug

run:
	wandb on;
	python ./main.py --multiple_gpu --wandb_name transformer

run_single_gpu:
	wandb on;
	python ./main.py --wandb_name transformer-single-gpu
