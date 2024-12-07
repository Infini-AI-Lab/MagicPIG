conda create -n magicpig python=3.10
conda activate magicpig
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install py-cpuinfo
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
pip install -r requirements.txt
pip install pytest
cd sparse_attention
pip install -e .
pytest test_sparse.py
cd ..
cd lsh
pip install -e .
cd ..
pytest test.py
