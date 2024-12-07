# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TEMPERATURE="0.0"
TOP_P="1.0"
TOP_K="32"
SEQ_LENGTHS=(
    16384
    32768
    65536
    98304
)

MODEL_SELECT() {
    MODEL_NAME=$1
    MODEL_DIR=$2
    ENGINE_DIR=$3
    
    case $MODEL_NAME in
        llama2-7b-chat)
            MODEL_PATH="meta-llama/Llama-2-7b-chat-hf"
            MODEL_TEMPLATE_TYPE="meta-chat"
            MODEL_FRAMEWORK="hf"
            ;;
        llama2-13b-chat)
            MODEL_PATH="meta-llama/Llama-2-13b-chat-hf"
            MODEL_TEMPLATE_TYPE="meta-chat"
            MODEL_FRAMEWORK="hf"
            ;;
        llama3-8b-chat)
            MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-chat"
            MODEL_FRAMEWORK="hf"
            ;;
        llama3-8b-chat-128k)
            MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-chat-3"
            MODEL_FRAMEWORK="hf"
            ;;
        llama3-70b-chat-128k)
            MODEL_PATH="meta-llama/Meta-Llama-3.1-70B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-chat-3"
            MODEL_FRAMEWORK="hf"
            ;;
        mistral-7b-chat-512k)
            MODEL_PATH="aws-prototyping/MegaBeam-Mistral-7B-512k"
            MODEL_TEMPLATE_TYPE="meta-chat"
            MODEL_FRAMEWORK="hf"
            ;;
        llama3-8b-chat-512k)
            MODEL_PATH="princeton-nlp/Llama-3-8B-ProLong-512k-Instruct"
            MODEL_TEMPLATE_TYPE="meta-chat-3"
            MODEL_FRAMEWORK="hf"
            ;;
    esac


    if [ -z "${TOKENIZER_PATH}" ]; then
        if [ -f ${MODEL_PATH}/tokenizer.model ]; then
            TOKENIZER_PATH=${MODEL_PATH}/tokenizer.model
            TOKENIZER_TYPE="nemo"
        else
            TOKENIZER_PATH=${MODEL_PATH}
            TOKENIZER_TYPE="hf"
        fi
    fi


    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:$MODEL_FRAMEWORK:$TOKENIZER_PATH:$TOKENIZER_TYPE:$OPENAI_API_KEY:$GEMINI_API_KEY:$AZURE_ID:$AZURE_SECRET:$AZURE_ENDPOINT"
}
