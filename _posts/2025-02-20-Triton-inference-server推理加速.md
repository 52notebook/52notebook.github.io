---
layout: post
title:  基于Triton-inference-server大模型推理加速
description: "  "
modified: 2025-02-20T10:50:15-08:00
tags: [LLM, 推理加速, TensorRT-LLM, Triton,  大模型] 


---

基于英伟达triton-inference-server部署大模型做推理加速。

<!-- more -->

Table of Contents
=================

* [背景](#背景)
   * [概念介绍](#概念介绍)
* [TensorRT-LLM容器](#tensorrt-llm容器)
* [TensorRT-Backend](#tensorrt-backend)
* [启动TensorRT-LLM容器](#启动tensorrt-llm容器)
* [HF模型转成TensorRT-LLM格式](#hf模型转成tensorrt-llm格式)
   * [模型仓库model_repo准备](#模型仓库model_repo准备)
   * [模型配置](#模型配置)
* [Triton启动服务](#triton启动服务)
   * [cURL验证服务](#curl验证服务)
   * [运行指标状态查看](#运行指标状态查看)
* [TensorRT-LLM模型编译不同参数配置推理性能调优](#tensorrt-llm模型编译不同参数配置推理性能调优)
* [Benchmark](#benchmark)
   * [使用Model Analyzer 分析模型性能](#使用model-analyzer-分析模型性能)
   * [tensorrt-backend](#tensorrt-backend-1)
   * [指标解读](#指标解读)
* [使用GEN-AI Perf分析模型性能](#使用gen-ai-perf分析模型性能)
* [其它](#其它)
   * [使用Triton CLI部署模型](#使用triton-cli部署模型)
      * [triton 客户端安装](#triton-客户端安装)
      * [triton模型转换](#triton模型转换)
* [引用](#引用)

# 背景

基于英伟达triton-inference-server部署大模型做推理加速。

## 概念介绍

TensorRT-LLM: 模型引擎

TensorRT-backend：推理后端。Triton-inference-server后端可选TensorRT-backend、vLLM、Python、Pytorch等。

TensorRT-Backend的角色是让你使用Triton-Inference-Server部署TensorRT-LLM模型。



# TensorRT-LLM容器

```shell
docker pull nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3
```

# TensorRT-Backend

triton 24.10  版本与TensorRT-Backend 0.14.0 是对齐关系。

```shell
git clone -b v0.14.0 https://github.com/triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend
git submodule update --init --recursive

如果网络不好，则修改.git/config和.gitmodules  。https://-->git://
```

# 启动TensorRT-LLM容器

```shell
docker run -itd --name triton-qwen \
--net host \
--shm-size=50g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--gpus '"device=4,5"' \
-v  /data/gpu/Project/tensorrtllm_backend:/tensorrtllm_backend \
-v /data/gpu/Project/engines:/engines \
-v /data/gpu/base_models:/models  \
nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3

docker exec -it triton-qwen  /bin/bash  
```

# HF模型转成TensorRT-LLM格式

```shell
GPTQ Int4量化首先安装下auto-gptq包

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip config set install.trusted-host mirrors.aliyun.com

cd /tensorrtllm_backend/tensorrt_llm/examples/qwen

# 注释掉requirements.txt中tensorrt-llm的安装（tritonserver镜像已包含对应版本）
pip install -r requirements.txt
pip install auto-gptq
                
                
#Build the Qwen-14B-Chat model using a single GPU and apply INT4 weight-only quantization.
python3 convert_checkpoint.py --model_dir /models/Qwen2-14B-Instruct/   \
                              --output_dir /engines/middle/Qwen2-14B-Instruct-Int4/  \
                              --dtype float16 \
                              --use_weight_only \
                              --weight_only_precision int4  \
                              --pp_size  2

trtllm-build --checkpoint_dir /engines/middle/Qwen2-14B-Instruct-Int4/ \
            --output_dir /engines/trt_engines/Qwen2-14B-Instruct-Int4/weight_only/2-gpu/ \
            --gemm_plugin float16

# With fp16 inference
mpirun -n 2 --allow-run-as-root  python3 ../run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len=50 \
                  --tokenizer_dir /models/Qwen2-14B-Instruct/ \
                  --engine_dir /engines/trt_engines/Qwen2-14B-Instruct-Int4/weight_only/2-gpu/
                  
  
```

## 模型仓库model_repo准备

```shell
mkdir -p /triton_model_repo/Qwen2-14B-Instruct-Int4-2Gpu 
cp -r /tensorrtllm_backend/all_models/inflight_batcher_llm/* /triton_model_repo/Qwen2-14B-Instruct-Int4-2Gpu 
```

## 模型配置

```shell
ENGINE_DIR=/engines/trt_engines/Qwen2-14B-Instruct-Int4/weight_only/2-gpu/
TOKENIZER_DIR=/models/Qwen2-14B-Instruct/
MODEL_FOLDER=/triton_model_repo/Qwen2-14B-Instruct-Int4-2Gpu 
TRITON_MAX_BATCH_SIZE=4
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=0
MAX_QUEUE_SIZE=0
FILL_TEMPLATE_SCRIPT=/tensorrtllm_backend/tools/fill_template.py
DECOUPLED_MODE=false

python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT},max_queue_size:${MAX_QUEUE_SIZE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT}
```

# Triton启动服务

```shell
# 'world_size' is the number of GPUs you want to use for serving. This should
# be aligned with the number of GPUs used to build the TensorRT-LLM engine.
python3 /tensorrtllm_backend/scripts/launch_triton_server.py --http_port=18000 --metrics_port=18002  --grpc_port=18001 --world_size=2 --model_repo=${MODEL_FOLDER}
```

![img](/images/2025/03/01_tensorrt_inference.png)

或使用tritonserver启动服务

```shell
-n 参数为 tp * pp

mpirun -n 2 --allow-run-as-root  tritonserver   --http-port=18000 --metrics-port=18002  --grpc-port=18001  --model-repo=${MODEL_FOLDER}
```

## cURL验证服务

```shell
curl -X POST localhost:18000/v2/models/ensemble/generate -d '{"text_input": "你好，你叫什么名字", "max_tokens": 100, "bad_words": "", "stop_words": ""}'
```

## 运行指标状态查看

```shell
curl localhost:18002/metrics
```

# TensorRT-LLM模型编译不同参数配置推理性能调优

```shell
#Build the Qwen-14B-Chat model using a single GPU and apply INT4 weight-only quantization.
python3 convert_checkpoint.py --model_dir /models/Qwen2-14B-Instruct/   \
                              --output_dir /engines/middle/Qwen2-14B-Instruct-Int4/  \
                              --dtype float16 \
                              --use_weight_only \
                              --weight_only_precision int4  \
                              --pp_size  2



# Build the engines
trtllm-build --checkpoint_dir /engines/middle/Qwen2-14B-Instruct-Int4/ \
             --output_dir /engines/trt_engines/Qwen2-14B-Instruct-Int4/weight_only/2-gpu-opti/ \
             --remove_input_padding enable \
             --gpt_attention_plugin float16 \
             --context_fmha enable \
             --gemm_plugin float16 \
             --kv_cache_type paged \
             --max_batch_size 64
             


mkdir -p /triton_model_repo/Qwen2-14B-Instruct-Int4-2Gpu-Opti

cp -r /tensorrtllm_backend/all_models/inflight_batcher_llm/* /triton_model_repo/Qwen2-14B-Instruct-Int4-2Gpu-Opti 



# 模型配置
# build模型存储路径
ENGINE_DIR=/engines/trt_engines/Qwen2-14B-Instruct-Int4/weight_only/2-gpu-opti/
TOKENIZER_DIR=/models/Qwen2-14B-Instruct/
# trt model repo仓库（四个模型文件路径）。模型仓库引用引擎模型地址。
MODEL_FOLDER=/triton_model_repo/Qwen2-14B-Instruct-Int4-2Gpu-Opti 
TRITON_MAX_BATCH_SIZE=64
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=0
MAX_QUEUE_SIZE=0
FILL_TEMPLATE_SCRIPT=/tensorrtllm_backend/tools/fill_template.py
DECOUPLED_MODE=false

# 新增
ACCUMULATE_TOKENS=true
MAX_BEAM_WIDTH=1
#MAX_TOKENS_IN_PAGED_KV_CACHE=${max_tokens_in_paged_kv_cache}
#MAX_ATTENTION_WINDOW_SIZE=${max_attention_windows_size}
#KV_CACHE_FREE_GPU_MEM_FRACTION=0.5
EXCLUDE_INPUT_IN_OUTPUT=true
ENABLE_KV_CACHE_REUSE=false


python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT},max_queue_size:${MAX_QUEUE_SIZE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},accumulate_tokens:${ACCUMULATE_TOKENS}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE},max_beam_width:${MAX_BEAM_WIDTH},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},enable_kv_cache_reuse:${ENABLE_KV_CACHE_REUSE}


# tritonserver服务启动
# -n 参数为 tp * pp
mpirun -n 2 --allow-run-as-root  tritonserver   --http-port=18000 --metrics-port=18002  --grpc-port=18001  --model-repo=${MODEL_FOLDER}
```

同样的model repo 中tensorrt-llm/config.pbtxt，但不一样的trtllm-build构建参数，benchmark吞吐性能差异很大。

| 构建指令参数                                                 | benchmark性能                                                | 备注                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| trtllm-build --checkpoint_dir /engines/middle/**Qwen2-14B-Instruct-Int4**/ \             --output_dir /engines/trt_engines/**Qwen2-14B-Instruct-Int4/**weight_only/2-gpu-opti/ \             --remove_input_padding enable \             --gpt_attention_plugin float16 \             --context_fmha enable \             --gemm_plugin float16 \             --kv_cache_type paged \             --max_batch_size 64 | INFO] Start benchmarking on 500 prompts.[INFO] Total Latency: 18953.21 ms[INFO] Total request latencies: 6050367.916999994 ms+----------------------------+----------+\|            Stat            \|  Value   \|+----------------------------+----------+\|        Requests/Sec        \|  26.38   \|\|       OP tokens/sec        \| 6753.56  \|\|     Avg. latency (ms)      \| 12100.74 \|\|      P99 latency (ms)      \| 18867.16 \|\|      P90 latency (ms)      \| 18863.80 \|\| Avg. IP tokens per request \|  256.00  \|\| Avg. OP tokens per request \|  256.00  \|\|   Avg. InFlight requests   \|   0.00   \|\|     Total latency (ms)     \| 18952.97 \|\|       Total requests       \|  500.00  \|+----------------------------+----------+ | 显卡占用70G 通过使用设置KV_CACHE_FREE_GPU_MEM_FRACTION=0.5约束显存占用，即便占用40G显存与70G的benchmark结果一致，说明推理服务并不是占用显存越高，性能越好；同时trtllm-build构建参数选择很重要。 |
| trtllm-build --checkpoint_dir /engines/middle/**Qwen2-14B-Instruct-Int4**/ \            --output_dir /engines/trt_engines/**Qwen2-14B-Instruct-Int4/**weight_only/2-gpu/ \            --gemm_plugin float16 | [INFO] Warm up for benchmarking.                                                                                                          [INFO] Start benchmarking on 500 prompts.                                                                                                 [INFO] Total Latency: 12249.455 ms                                                                                                        [INFO] Total request latencies: 6073121.504000006 ms                                                                                      +----------------------------+----------+                                                                                                 \|            Stat            \|  Value   \|                                                                                                 +----------------------------+----------+                                                                                                 \|        Requests/Sec        \|  40.82   \|                                                                                                 \|       OP tokens/sec        \| 10449.64 \|                                                                                                 \|     Avg. latency (ms)      \| 12146.24 \|                                                                                                 \|      P99 latency (ms)      \| 12171.10 \|                                                                                                 \|      P90 latency (ms)      \| 12168.66 \|                                                                                                 \| Avg. IP tokens per request \|  256.00  \|                                                                                                 \| Avg. OP tokens per request \|  256.00  \|                                                                                                 \|   Avg. InFlight requests   \|   0.00   \|                                                                                                 \|     Total latency (ms)     \| 12249.22 \|                                                                                                 \|       Total requests       \|  500.00  \|                                                                                                 +----------------------------+----------+ | 显卡占用70G                                                  |

# Benchmark

## 使用Model Analyzer 分析模型性能

```shell
pip3 install triton-model-analyzer
```

## tensorrt-backend

生成具有输入正态分布的 I/O seqlen 令牌，mean_seqlen=128，stdev=10。输出正态分布，mean_seqlen=20，stdev=2。设置 stdev=0 以获得常数 seqlens。

```shell
python3 /tensorrtllm_backend/tools/inflight_batcher_llm/benchmark_core_model.py  --num-requests 500 --max-input-len 1024 -i grpc token-norm-dist --input-mean 256 --input-stdev 0 --output-mean 256 --output-stdev 0 
```

性能指标输出结果

```shell
[INFO] Warm up for benchmarking.                                                                                                         
[INFO] Start benchmarking on 500 prompts.                                                                                                
[INFO] Total Latency: 12249.455 ms                                                                                                       
[INFO] Total request latencies: 6073121.504000006 ms                                                                                     
+----------------------------+----------+                                                                                                
|            Stat            |  Value   |                                                                                                
+----------------------------+----------+                                                                                                
|        Requests/Sec        |  40.82   |                                                                                                
|       OP tokens/sec        | 10449.64 |                                                                                                
|     Avg. latency (ms)      | 12146.24 |                                                                                                
|      P99 latency (ms)      | 12171.10 |                                                                                                
|      P90 latency (ms)      | 12168.66 |                                                                                                
| Avg. IP tokens per request |  256.00  |                                                                                                
| Avg. OP tokens per request |  256.00  |                                                                                                
|   Avg. InFlight requests   |   0.00   |                                                                                                
|     Total latency (ms)     | 12249.22 |                                                                                                
|       Total requests       |  500.00  |                                                                                                
+----------------------------+----------+ 
```

## 指标解读

模拟场景：发送500次请求，请求输入最长序列长度为1024，均值为256，输出序列长度均值256 token。

1. **Total Latency**: 总延迟时间，表示完成所有请求所需的总时间。在这个例子中，总延迟是12249.455毫秒（ms）。
2. **Total request latencies**: 所有请求的延迟时间总和。这里给出的是6073121.504毫秒。这个值应该与 `Total Latency` 相同，因为它是所有请求延迟的累加。
3. **Requests/Sec**: 请求每秒（Requests per Second），表示平均每秒处理的请求数量。这里显示的是40.82，意味着平均每秒处理约40个请求。
4. **OP tokens/sec**: 输出令牌每秒（Output tokens per second），表示平均每秒生成的输出令牌数量。这里显示的是10449.64，意味着平均每秒生成约10449个输出令牌。
5. **Avg. latency (ms)**: 平均延迟（Average Latency），表示所有请求的平均延迟时间。这里显示的是12146.24毫秒。
6. **P99 latency (ms)**: 第99百分位延迟，表示99%的请求延迟时间不超过这个值。这里显示的是12171.10毫秒。
7. **P90 latency (ms)**: 第90百分位延迟，表示90%的请求延迟时间不超过这个值。这里显示的是12168.66毫秒。
8. **Avg. IP tokens per request**: 每个请求的平均输入令牌数（Average Input tokens per request），表示每个请求平均处理的输入令牌数量。这里显示的是256.00。
9. **Avg. OP tokens per request**: 每个请求的平均输出令牌数（Average Output tokens per request），表示每个请求平均生成的输出令牌数量。这里显示的是256.00。
10. **Avg. InFlight requests**: 平均在飞请求数（Average In-Flight requests），表示在测试期间任何时刻平均正在处理的请求数量。这里显示的是0.00，意味着在测试期间没有并发请求。
11. **Total latency (ms)**: 总延迟时间，与第一个指标相同，这里再次列出以供参考，显示的是12249.22毫秒。
12. **Total requests**: 总请求数，表示在测试期间处理的请求总数。这里显示的是500.00。

# 使用GEN-AI Perf分析模型性能

在Triton容器内，执行下面命令

英伟达自构建的Triton容器内genai-perf版本较老，可能会无法执行评估任务。升级版本：

```shell
#https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
pip install git+https://github.com/triton-inference-server/perf_analyzer.git#subdirectory=genai-perf
genai-perf \
  profile \
  -m ensemble \
  --service-kind triton \
  --backend tensorrtllm \
  --num-prompts 1000 \
  --random-seed 123 \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --output-tokens-mean-deterministic \
  --tokenizer /models/Qwen2-14B-Instruct/ \
  --concurrency 80 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:18001  \
  --server-metrics-url  http://localhost:18002/metrics
  
  
  或 
 triton profile -m ensemble  --backend tensorrtllm
```



![img](https://rv6zltn2r6e.feishu.cn/space/api/box/stream/download/asynccode/?code=ODVkM2UyNzYxNTE4ZjA4NGM4NmU4Nzk4NzVlNGExZTVfSFdGWEVPMm1NYnpEQnhxUTk4QkhWWVY2cmhkOHY0d05fVG9rZW46U2k3V2JkbHhPb012WHl4RVJUemNYMzc1bnpsXzE3NDI1MTkwMjI6MTc0MjUyMjYyMl9WNA)

# 其它

## 使用Triton CLI部署模型

### triton 客户端安装

```shell
# 查看最新版本号 https://github.com/triton-inference-server/triton_cli/releases

GIT_REF=0.0.11
pip install git+https://github.com/triton-inference-server/triton_cli.git@${GIT_REF}
```

### triton模型转换

```shell
ENGINE_DEST_PATH=/engines triton import -m llama-2-7B --backend tensorrtllm
```



# 引用：

> HF模型Python Backend部署
>
> https://github.com/triton-inference-server/tutorials/blob/main/Quick_Deploy/HuggingFaceTransformers/README.md
>
> triton部署千问示例  
>
> https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/qwen/README.md
>
> 性能分析
>
> https://github.com/triton-inference-server/tutorials/blob/main/Quick_Deploy/HuggingFaceTransformers/README.md
>
> https://github.com/triton-inference-server/perf_analyzer
>
> https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/perf-best-practices.md
>
> 性能优化
>
> https://tensorrt-llm.continuumlabs.ai/best-practices-for-tuning-the-performance-of-tensorrt-llm
>
> https://github.com/triton-inference-server/tutorials/blob/main/Popular_Models_Guide/Llama2/trtllm_guide.md



