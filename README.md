# MSrag


Start: 
uvicorn main:app --host localhost --port 7000 --reload


docker run --runtime nvidia --gpus all \
    --name gemma12b-qat \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=huggingface" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model gaunernst/gemma-3-12b-it-qat-compressed-tensors


docker run --runtime nvidia --gpus all \
    vllm/vllm-openai:latest \
    --model gaunernst/gemma-3-12b-it-qat-compressed-tensors \
    --max-model-len 16000  \
    --gpu-memory-utilization 0.5  