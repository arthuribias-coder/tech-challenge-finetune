# Tech Challenge 3 - Fine-Tuning Foundation Model

Projeto de fine-tuning de um foundation model utilizando o dataset AmazonTitles-1.3MM para geração de descrições de produtos a partir de títulos.

## Objetivo

Executar fine-tuning de um foundation model (Llama, BERT, etc.) para que, dado o título de um produto, o modelo seja capaz de gerar sua descrição baseado no aprendizado obtido do dataset AmazonTitles-1.3MM.

## Estrutura do Projeto

```
.
├── dataset/                          # Dataset AmazonTitles-1.3MM (trn.json)
├── docs/                             # Documentação do projeto
│   ├── tech-challenge.md            # Enunciado original
│   └── plano-implementacao.md       # Plano detalhado de implementação
├── tech_challenge_finetune.ipynb    # Notebook principal com todo o processo
├── requirements.txt                  # Dependências Python
├── README.md                         # Este arquivo
└── .gitignore                        # Arquivos ignorados pelo Git
```

## Pré-requisitos

- Python 3.9+
- CUDA (recomendado para GPU)
- 16GB+ RAM
- 10GB+ espaço em disco

## Instalação

1. Clone o repositório:
```bash
git clone <URL_DO_REPOSITORIO>
cd TC-3
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Baixe o dataset:
- Download: [AmazonTitles-1.3MM](https://drive.google.com/file/d/12zH4mL2RX8iSvH0VCNnd3QxO4DzuHWnK/view)
- Extraia o arquivo `trn.json` para o diretório `dataset/`

## Uso

1. Abra o Jupyter Notebook:
```bash
jupyter notebook tech_challenge_finetune.ipynb
```

2. Execute as células sequencialmente seguindo as instruções inline

## Fluxo de Trabalho

1. **Preparação do Dataset**: Carregamento e processamento do trn.json
2. **Modelo Base**: Teste do modelo antes do fine-tuning
3. **Fine-Tuning**: Treinamento com LoRA/PEFT
4. **Avaliação**: Comparação antes/depois do fine-tuning
5. **Demonstração**: Interface para testes interativos

## Tecnologias Utilizadas

- **PyTorch**: Framework de deep learning
- **Transformers (Hugging Face)**: Modelos foundation
- **PEFT**: Parameter-Efficient Fine-Tuning (LoRA)
- **Pandas**: Manipulação de dados
- **Datasets**: Processamento de datasets

## Modelo Foundation

**Opção escolhida**: A ser definido durante implementação
- Llama 3.2 (1B ou 3B)
- BERT-base
- DistilBERT

## Entregáveis

- [ ] Código-fonte (este repositório)
- [ ] Vídeo demonstrativo (máx. 10 minutos)
- [ ] PDF com links (YouTube + GitHub)

## Licença

Projeto acadêmico - Pós-Graduação

## Contato

Desenvolvido para o Tech Challenge - Fase 3
