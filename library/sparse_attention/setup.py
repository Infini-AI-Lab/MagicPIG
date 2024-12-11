from torch.utils.cpp_extension import CppExtension
import setuptools

import cpuinfo
import subprocess
def supports_avx512_bf16():
    """Check if the CPU supports AVX512_BF16."""
    try:
        info = cpuinfo.get_cpu_info()
        return "avx512_bf16" in info.get("flags", [])
    except Exception:
        return False

def is_gcc_version_greater_than_11():
    try:
        # 获取 gcc 版本信息
        result = subprocess.run(["gcc", "-dumpversion"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            # 提取版本号
            version_str = result.stdout.strip()
            major_version = int(version_str.split('.')[0])  # 获取主版本号
            return major_version > 10
        else:
            print("GCC is not installed or not found in the PATH.")
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Check for AVX512_BF16 support
avx_flags = []
if supports_avx512_bf16() and is_gcc_version_greater_than_11():
    avx_flags.extend(["-mavx512bf16"])
# Define the extension modules
ext_modules = [
    CppExtension(
        "sparse_attention_cpu",
        sources=[
            "sparse_attention.cc",
            "3rdparty/FBGEMM/src/FbgemmBfloat16Convert.cc",
            "3rdparty/FBGEMM/src/FbgemmBfloat16ConvertAvx2.cc",
            "3rdparty/FBGEMM/src/FbgemmBfloat16ConvertAvx512.cc",
            "3rdparty/FBGEMM/src/RefImplementations.cc",
            "3rdparty/FBGEMM/src/Utils.cc",
        ],
        include_dirs=["3rdparty/FBGEMM/include"],
        extra_compile_args=avx_flags + ["-mavx512f", "-fopenmp", "-D_GLIBCXX_USE_CXX11_ABI=0", "-std=c++17"],
        extra_link_args=['-fopenmp'],
    ),
]

setuptools.setup(name="sparse_attention_cpu", version="0.1.0", ext_modules=ext_modules)