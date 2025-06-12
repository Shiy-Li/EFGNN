# EFGNN
[TNNLS-2025] Uncertainty-Aware Graph Neural Networks: A Multi-Hop Evidence Fusion Approach

# Setup
```js/java/c#/text
conda create -n EFGNN python=3.8
conda activate EFGNN
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.2+cu121.html
pip install torch-geometric==2.6.1

```

# Usage
Just run the script corresponding to the dataset and method you want. For instance:

```js/java/c#/text
python main.py --dataset cora
```

# Cite
If you compare with, build on, or use aspects of this work, please cite the following:

```js/java/c#/text
@inproceedings{chen2025uncertainty,
  title={Uncertainty-Aware Graph Neural Networks: A Multi-Hop Evidence Fusion Approach},
  author={Chen, Qingfeng and Li, Shiyuan and Liu, Yixin and Pan, Shirui and Geoff, Webb and Zhang, Shichao},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025}
}
```

