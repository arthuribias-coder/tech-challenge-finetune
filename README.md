# Tech Challenge 3 - Fine-Tuning Foundation Model

Fine-tuning do modelo **Phi-4-mini-instruct** (3.8B parâmetros) com **LoRA** no dataset **AmazonTitles-1.3MM** para geração de descrições de produtos a partir de títulos.

**Resultado**: Dado um título de produto, o modelo gera descrições detalhadas no estilo Amazon.

## Arquitetura

**Modelo**: Phi-4-mini-instruct (Microsoft, 3.8B parâmetros)  
**Técnica**: LoRA - treina apenas 0.08% dos parâmetros (3.1M/3.8B)  
**Dataset**: AmazonTitles-1.3MM - 3K amostras (2.7K treino / 300 validação)  
**Hardware**: NVIDIA RTX 2000 Ada (16GB)  
**Treinamento**: 2 épocas, ~60 minutos, loss 2.89→2.77

```
Dados Brutos (2.2M) → Limpeza (1.3M) → Amostra (3K) → Tokenização → LoRA Fine-Tuning → Modelo Especializado
```## Instalação Rápida

```bash
git clone https://github.com/arthuribias-coder/tech-challenge-finetune.git
cd tech-challenge-finetune
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Dataset**: [Download AmazonTitles-1.3MM](https://drive.google.com/file/d/12zH4mL2RX8iSvH0VCNnd3QxO4DzuHWnK/view) → Extrair `trn.json` para `dataset/`

**Requisitos**: Python 3.9+, GPU NVIDIA 12GB+, 16GB RAM

## Como Usar

**Treinamento completo** (~60 min):

```bash
jupyter notebook tech_challenge_finetune.ipynb
# Execute as células em ordem (1-32)
```

**Apenas inferência** (modelo já treinado):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./modelo_finetuned")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")

prompt = "Descreva o produto: Wireless Bluetooth Headphones\n\nDescrição:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## Resultados

**Métricas**: Loss 2.891 → 2.772 (4.1% redução) | 2 épocas (~60 min) | 13 checkpoints | 3.1M parâmetros (0.08%)

**Antes**: ❌ Genérico | ❌ Alucinações | ❌ Irrelevante  
**Depois**: ✅ Conciso | ✅ E-commerce style | ✅ Redução alucinações

**Exemplo**:

```
Input: "Wireless Bluetooth Headphones with Noise Cancellation"

Baseline: "A massa de Goodness Gracious Hula Lula Turkey 5 oz..." (confuso)
Fine-tuned: "Premium wireless headphones featuring active noise cancellation..." (relevante)
```

## Tecnologias

- **PyTorch 2.8.0** + **Transformers 4.49.0** (Hugging Face)
- **PEFT 0.17.1** (LoRA)
- **Accelerate 1.10.1** (otimização)
- **Datasets 4.1.1** + **Pandas 2.3.3**

## Troubleshooting

**CUDA out of memory**: Reduza `BATCH_SIZE=1` ou `MAX_LENGTH=64`  
**Treinamento lento**: Verifique GPU com `torch.cuda.is_available()`  
**Tokenizer warning**: `os.environ["TOKENIZERS_PARALLELISM"] = "false"`

## Referências

- [Phi-4 Model Card](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [AmazonTitles Dataset](https://github.com/xuyige/SIGIR2019-One2Set)

---

**Tech Challenge - Fase 3** | Pós-Graduação em Inteligência Artificial
