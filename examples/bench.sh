numactl -C 0-31,52-83 -m 0,1 python bench.py --B 1 --K 0 --L 150 --model codellama/CodeLlama-7b-Instruct-hf --M 16384 --P 16000
numactl -C 0-31,52-83 -m 0,1 python bench.py --B 4 --K 0 --L 150 --model codellama/CodeLlama-7b-Instruct-hf --M 16384 --P 16000
numactl -C 0-31,52-83 -m 0,1 python bench.py --B 8 --K 0 --L 150 --model codellama/CodeLlama-7b-Instruct-hf --M 16384 --P 16000
# numactl -C 0-31,52-83 -m 0,1 python bench.py --B 1 --K 10 --L 170 --model codellama/CodeLlama-7b-Instruct-hf --M 131072 --P 128000


# numactl -C 0-31,52-83 -m 0,1 python bench.py --B 12 --K 9 --L 120 --model codellama/CodeLlama-7b-Instruct-hf --M 131072 --P 128000


# numactl -C 0-31,52-83 -m 0,1 python bench.py --B 12 --K 8 --L 75 --model codellama/CodeLlama-7b-Instruct-hf --M 131072 --P 128000
