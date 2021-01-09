debug:
	wandb off;
	python ./main.py --max_seq_len 50 --batch_size 2 --debug

run:
	wandb on;
	python ./main.py --max_seq_len 50 --multiple_gpu --wandb_name transformer

run_single_gpu:
	wandb on;
	python ./main.py --max_seq_len 50 --wandb_name transformer-single-gpu
