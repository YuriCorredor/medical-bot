# Steps to run

```bash
conda create -n mchatbot python=3.8 -y
conda activate mchatbot
```

```bash
pip install -r requirements.txt
```

```bash
conda deactivate
```

```bash
conda remove -n mchatbot --all -y
```

Download model from [here](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q4_0.bin) and place it in the model folder.
