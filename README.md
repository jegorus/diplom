# Study of methods for fine-tuning small-sized LLM's (Исследование методов дообучения языковых моделей малого размера)

## Problem Statement:
1) A small-sized language LLM model (BERT) is given (about 100M parameters)
2) A set of datasets for fine-tuning
3) Several fine-tuning methods that will need to be compared in terms of quality, time and used GPU memory + get reproducible results

## To run code: 
open diplom_runner_final.ipynb in Google Colab or Jupyter notebook. Run first 4 cells. First will clone repo in colab and install dependencies, second will choose method and dataset, in third you need to put training_args (dict), and fourth will run the code

### Repo description: 
- diplom_runner_final.ipynb - use it to run experiments
- notebooks: Old and additional notebooks in google colab.
- results: contains raw files of experiments and results. Check README for better interpretation
- scripts: dependencies and logs clearing
- utils: utils: contains Custom Trainer and functions to print parameters
- files *.py will get cloned in the first colab cell (automatically):
  - model.py: ModelHandler to run experiments
  - data.py: DatasetHandler to process datasets
  - config.py: use fields from it to configure fine-tuning method
  - convert.py: is used to convert data into tables and graphs

### Model: [RoBERTa](https://arxiv.org/abs/1907.11692)
### Datasets: Chosen from [GLUE](https://gluebenchmark.com/) and [SUPERGLUE](https://super.gluebenchmark.com/)
1) mrpc (paraphrase, glue, 5.8k)
2) cb (entailment, superglue, 556) 
3) boolq (QA, superglue, 15.9k)
4) rte (entaliment, superglue, 5.77k)
### Methods:
1) Full fune-tuning
2) [LoRA](https://arxiv.org/abs/2106.09685)
3) [IA3](https://arxiv.org/abs/2205.05638)
4) [AdaLoRA](https://arxiv.org/abs/2303.10512)

## Results:
Check the presentation and paper (russian) from thesis folder for better understanding
### Learning Rate
The best learning rate for the task was found:
- FullFt: 2e-5, 5e-5
- LoRa: 5e-4, 7e-4
- IA3: 5e-3
- AdaLora: 1e-3, 2e-3
<img src="https://github.com/user-attachments/assets/02896938-93a0-46bc-b298-73b38787b51d" width=50%>

### Alpha and rank LoRA:
Hypotheses to test:
1) r = alpha, r = 2alpha: [lora paper](https://arxiv.org/abs/2106.09685)
2) r = 2alpha works best: [microsoft](https://github.com/microsoft/LoRA) + [lightning AI](https://lightning.ai/pages/community/lora-insights/) Llama 7B
3) "We find LoRA r is unrelated to final performance" - [qlora paper](https://arxiv.org/abs/2305.14314), where r >= 8, alpha = big (=64)
4) r > alpha works good: [qlora](https://arxiv.org/abs/2305.14314) for big models: >= 7B 


results:
- with increasing rank and alpha the results become better
- small rank works relatively well
- both of them have approximately the same effect on the result (alpha is a little stronger)
- don't allow big difference between them
- All the hypotheses like: a=r, a=2r: don't make a big difference
<img src="https://github.com/user-attachments/assets/199cb401-ec78-4780-b04a-c670d0cc0661" width=50%>

### Target modules LoRA
Hypotheses: 
1) “We limit our study to only adapting the attention weights for downstream tasks and freeze the MLP modules (so they are not trained in downstream tasks) both for simplicity and parameter-efficiency” – [Lora paper](https://arxiv.org/abs/2106.09685)
2) all layers give the best results - [Qlora paper](https://arxiv.org/abs/2305.14314)

results: 
1) The modules that have the greatest impact on the result are: value and intermediate ff
2) On average, the more modules to which the lora is applied, the better the result
3) We recommend using modules in MLP and self-attention output (so, the best results for two modules are: one module was from self-attention and one from MLP) \
image description: mrpc: seed=42, 1, 2, rte: seed=5, modules: qvk, qv, vk, v
<img src="https://github.com/user-attachments/assets/8edc2e3d-42f4-4f18-90b4-06e04d5187d3" width=50%>

### Fine-tuning speed dependence from rank and alpha and modules:
1) All modules - about slower 25%
2) Alpha – doesn't change
3) Rank – less than 10% slower
<img src="https://github.com/user-attachments/assets/b630a3dd-53f6-48ca-a82f-8f3655262b5d" width=50%>

### Ia3 modules results:
1) qvk, qv, vk – among the modules query, value, key - the best
2) intff-outff and attff-outff – among the rest - the best
3) adding layers to feedforward slightly improves the result
4) adding modules reduces the speed by no more than 10%

### Adalora modules results:
1) All-layers – shows the best results
2) Results are slightly higher than LoRa
3) The most significant impact gives value and intff
4) Adding all modules reduces the learning speed to 50%
<img src="https://github.com/user-attachments/assets/4a580689-f742-4117-8b8f-20178e8a7fdf" width=50%>

## Comparison:
### GPU
1) PEFT: 2/3 of full fine-tuning
2) Adapters with all modules have similar GPU load with full fine-tuning
3) Rank does not affect the GPU load much
<img src="https://github.com/user-attachments/assets/03c346eb-1ceb-4d52-b880-44447c5055b0" width=50%>

### Accuracy and speed
<img src="https://github.com/user-attachments/assets/a3265d99-6721-4bfe-aa3d-38ec9650209d" width=50%>

## Conclusions
1) AdaLoRa, LoRa show the best results
Using the t-criterion: LoRa > FullFt – stat. significant
AdaLora > Lora – did not receive stat. significance
2) Most time: full fine-tuning and AdaLoRa
least time – ia3 and lora on 1-2 modules. Up to 1.5 times less
3) GPU load: 2/3 of full fine-tuning
4) the best results (LoRA, AdaLoRA) rank: 8 – 64, alpha ≈ rank
5) modules: value and intff – optimal time/quality, all-layers – best results




















