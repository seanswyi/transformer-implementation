debug:
	wandb off;
	python ./main.py --max_seq_len 50 --batch_size 1 --debug


run:
	python ./main.py --max_seq_len 50 --multiple_gpu --wandb_name transformer

run_single_gpu:
	python ./main.py --max_seq_len 50 --wandb_name transformer-single-gpu
