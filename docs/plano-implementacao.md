# Plano de Implementação - Tech Challenge

## Proposta: Jupyter Notebook Único

**Recomendação:** Um único Jupyter Notebook seria a melhor opção para este projeto, pois:

- Permite documentar cada etapa inline com Markdown
- Facilita visualização de resultados intermediários
- Simplifica a entrega e apresentação no vídeo
- Mantém todo o fluxo de trabalho em um único arquivo

## Estrutura Proposta

### Arquivo único: `tech_challenge_finetune.ipynb`

**Seções do Notebook:**

### 1. Configuração Inicial

- Instalação de dependências (`transformers`, `torch`, `pandas`, `datasets`, `peft`)
- Imports necessários
- Configuração de variáveis globais (paths, parâmetros)

### 2. Preparação do Dataset

- Carregar `trn.json` com pandas
- Análise exploratória básica (quantidade de registros, exemplos)
- Limpeza de dados (remover nulos, duplicatas)
- Formatação dos dados para fine-tuning no padrão prompt-completion:
  - **Prompt:** "Descreva o produto: {title}"
  - **Completion:** "{content}"
- Split treino/validação (90/10 ou 95/5)
- Salvar subset processado (trabalhar com 10-50k registros para viabilidade)

### 3. Teste do Modelo Base (Pré Fine-Tuning)

- Carregar modelo foundation (sugestão: **Llama 3.2 3B** via Ollama ou **BERT-base**)
- Realizar 3-5 testes com títulos do dataset
- Documentar respostas antes do treinamento

### 4. Fine-Tuning

**Abordagem simplificada com PEFT/LoRA:**

- Configurar LoRA (Low-Rank Adaptation) para fine-tuning eficiente
- Parâmetros reduzidos para execução local:
  - Epochs: 1-3
  - Batch size: 4-8
  - Learning rate: 2e-4
  - LoRA rank: 8-16
- Treinar apenas com subset do dataset (10k-50k samples)
- Salvar adapter/checkpoint

### 5. Avaliação e Comparação

- Carregar modelo fine-tunado
- Testar com os mesmos prompts da seção 3
- Comparar respostas antes/depois
- Métricas qualitativas (coerência, relevância)

### 6. Interface de Demonstração

- Função simples para fazer perguntas
- Loop interativo ou exemplos fixos para o vídeo
- Demonstração de 3-5 casos de uso

## Stack Tecnológica Minimalista

```
bibliotecas principais:
- pandas (manipulação de dados)
- transformers (Hugging Face)
- peft (Parameter-Efficient Fine-Tuning)
- torch (backend)
- datasets (processamento)
```

## Modelo Recomendado

### Opção 1 (Mais Simples)

- **BERT-base** ou **DistilBERT** para classificação/geração de descrições
- Mais leve, rápido para treinar
- Focado em entendimento de texto

### Opção 2 (Mais Alinhada com Enunciado)

- **Llama 3.2 3B** (ou 1B) via Transformers
- Modelo generativo moderno
- Requer mais recursos mas resultados melhores

## Estratégia de Simplificação

Para manter simples e viável:

1. **Dataset reduzido:** Usar apenas 10k-50k amostras do trn.json
2. **LoRA Fine-Tuning:** Treinar apenas adapters (não o modelo completo)
3. **Sem infraestrutura complexa:** Tudo local ou Google Colab
4. **Sem API/Deploy:** Apenas demonstração no notebook
5. **Documentação inline:** Markdown explicativo entre células de código

## Entregáveis

- `tech_challenge_finetune.ipynb` (notebook completo)
- `README.md` (instruções de execução)
- `requirements.txt` (dependências)
- Opcional: `modelo_finetuned/` (se for pequeno o suficiente)

## Timeline Sugerido

1. **Preparação dados:** 2-3 horas
2. **Setup e teste modelo base:** 1-2 horas
3. **Fine-tuning:** 3-6 horas (dependendo do hardware)
4. **Avaliação e ajustes:** 2-3 horas
5. **Documentação e vídeo:** 3-4 horas

**Total:** 15-20 horas de trabalho

## Vantagens desta Abordagem

- Tudo em um único arquivo navegável
- Fácil de apresentar no vídeo
- Reproduzível (com requirements.txt)
- Documentação integrada ao código
- Baixa complexidade de arquitetura
- Focado no core: preparação → fine-tuning → demonstração

Esta abordagem atende todos os requisitos do enunciado mantendo máxima simplicidade.
