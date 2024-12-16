numactl -C 0-31,52-83 -m 0,1 python bench.py --B 1 --K 10 --L 150 --model meta-llama/Meta-Llama-3.1-8B-Instruct --M 98304 --P 98000
