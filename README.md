# **Multi-Attribute Steering of Language Models via Targeted Interventions**

[Duy Nguyen](https://duykhuongnguyen.github.io/), [Archiki Prasad](https://archiki.github.io/), [Elias Stengel-Eskin](https://esteng.github.io/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/).

![image](assets/mat_steer_fig.png)

## **Overview**
This repository provides the implementation for **Multi-Attribute Steering of Language Models via Targeted Interventions (MAT-Steer)**. MAT-Steer enables selective token-level intervention across multiple attributes for language models.

## **Installation**

1. **Set up the environment**:
   ```bash
   conda env create -f environment.yaml
   conda activate iti
   python -m ipykernel install --user --name iti --display-name "iti"
   ```

2. **Create necessary directories**:
   ```bash
   mkdir -p validation/results_dump/summary_dump/test 
   mkdir -p validation/results_dump/summary_dump/val
   mkdir -p validation/answer_dump/summary_dump/test
   mkdir -p validation/answer_dump/summary_dump/val
   ```

## **Running MAT-Steer**

### **1. Extract Model Activations**
Navigate to the `get_activations` directory and run:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python get_activations.py --model_name llama3.1_8B --dataset_name truthfulqa
   CUDA_VISIBLE_DEVICES=0 python get_activations.py --model_name llama3.1_8B --dataset_name toxigen
   CUDA_VISIBLE_DEVICES=0 python get_activations.py --model_name llama3.1_8B --dataset_name bbq
   ```

### **2. Train Steering Vectors**
Navigate to the `validation` directory. Running using the default setting:
   ```bash
   python steering.py \
    --model_name llama3.1_8B \
    --batch_size 96 \
    --epochs 100 \
    --lr 0.001 \
    --sigma 2.0 \
    --lambda_mmd 1.0 \
    --lambda_sparse 0.9 \
    --lambda_ortho 0.1 \
    --lambda_pos 0.9
   ```

### **3. Apply Targeted Intervention**
Then the steering vectors can be integrated into models following [ITI](https://github.com/likenneth/honest_llama) and [pyvene](https://github.com/stanfordnlp/pyvene). The entire code on steering and evaluation will be releasing soon.

## **Citations**
If you find this work useful, please consider citing our paper:
    ```bash
    @article{nguyen2025multi,
        title={Multi-Attribute Steering of Language Models via Targeted Intervention},
        author={Nguyen, Duy and Prasad, Archiki and Stengel-Eskin, Elias and Bansal, Mohit},
        journal={arXiv preprint arXiv:2502.12446},
        year={2025}
    }
    ```
