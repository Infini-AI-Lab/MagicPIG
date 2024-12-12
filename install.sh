pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install py-cpuinfo
pip install Cython
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
pip install -r requirements.txt
pip install pytest
cd library
cd sparse_attention
pip install -e .
pytest test_sparse.py
cd ..
cd lsh
pip install -e .
pytest test.py
cd ../..