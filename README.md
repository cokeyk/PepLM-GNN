# PepLM-GNN

<div align="center">

</div>

The precise prediction of peptide-protein interaction (PepPI) is a core support for promoting breakthroughs in peptide drug research, as well as understanding the regulatory mechanisms of biomolecules. Reserachers have developed several computational methods to predict PepPI. However, existing computational methods also have significant limitations.Inthis study, we propose a computing framework, PepLM-GNN, that integrates pre-trained language ProtT5 model with hybrid graph network for accurate identification of PepPI. This model constructs a graph by using ProtT5-extracted semantic context features of peptides and proteins to form heterogeneous nodes, with edges connecting interacting peptide-protein pairs. The hybrid graph network utilizes Graph Convolutional Networks (GCN) to provide the comprehensive information of the peptide and protein sequences, while employing the Graph Isomorphism Network (GIN) to capture the global interations between them.Compared with the existing advanced methods, PepLM-GNN demonsited the high accurately performance and robustness in predicting the PepPIs. We further demonstrated the capabilities of PepLM-GNN in virtual peptide drug screening, which is expected to facilitate the discovery of peptide drugs and the elucidation of protein functions.

**To avoid environment configuration complexities and ensure stable usage, we strongly recommend using the PepLM-GNN online prediction server. Access it at:http://bliulab.net/PepLM-GNN/.**

![Model](/imgs/Model.png)

**Fig. 1: The framework of PepLM-GNN, comprising four modules: ProtT5, graph convolution, graph isomorphism, and classification.** 

# 1 Installation

## 1.1 Create conda environment

```
conda create -n peplm-gnn python=3.10
conda activate peplm-gnn
```

## 1.2 Requirements
We recommend installing the environment using the provided `requirements.txt` file to ensure compatibility:
```
pip install -r requirements.txt
```

> **Note** If you have an available GPU, the accelerated PepLM-GNN can be used to predict peptide-protein binary interactions. Change the URL below to reflect your version of the cuda toolkit (cu118 for cuda=11.6 and cuda 11.8, cu121 for cuda 12.1). However, do not provide a number greater than your installed cuda toolkit version!
> 
> ```
> pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
> ```
>
> For more information on other cuda versions, see the [pytorch installation documentation](https://pytorch.org/).

## 1.3 Tools
Feature extraction tools and databases on which PepLM-GNN relies: 
```
ProtT5 
```

**The default paths to all tools and databases are shown in `conf.py`. You can change the paths to the tools and databases as needed by configuring `conf.py`.**

`ProtT5` are recommended to be configured as the system envirenment path. Your can follow these steps to install it:

### 1.3.1 How to install ProtT5 (ProtT5-XL-UniRef50)
Download and install (More information, please see **https://github.com/agemagician/ProtTrans** or **https://zenodo.org/record/4644188**, about 5.3GB)

```
wget https://zenodo.org/records/4644188/files/prot_t5_xl_uniref50.zip?download=1
unzip prot_t5_xl_uniref50.zip
```

## 1.4 Install PepLM-GNN
To install from the development branch run
```
git clone git@github.com:cokeyk/PepLM-GNN.git
cd PepLM-GNN/
```

**Finally, configure the defalut path of the above tool and the database in `conf.py`. You can change the path of the tool and database by configuring `conf.py` as needed.**


# 2 Usage
It takes 2 steps to predict peptide-protein binary interaction:

(1) Replace the default peptide sequence in the `example/Peptide_Seq.fasta` file with your peptide sequence (FASTA format). Similarly, replace the default protein sequence in the `example/Protein_Seq.fasta` file with your protein sequence (FASTA format). If you don't want to do this, you can also test your own peptide-protein pairs by modifying the paths to the files passed in by the `run_predictor.py` script (the parameter is `-uip`, respectively).

(2) Next, run `run_predictor.py` to make predictions. It should be noted that `run_predictor.py` automatically calls the script `FeatureExtract.py` to generate features for peptides and proteins.
```
conda activate peplm-gnn
python run_predictor.py -uip example
```

If you want to retrain based on your private dataset, find the original PepLM-GNN model in `model.py`. The PepLM-GNN source code we wrote is based on the Pytorch implementation and can be easily imported by instantiating it.

# 3 Problem feedback
If you identify any potential bugs, feel free to raise them in the [issuetracker](https://github.com/cokeyk/PepLM-GNN/issues).

In addition, if you have any further questions about PepLM-GNN, please feel free to contact us [**kyan@bliulab.net**]

# 4 Citation

If you find our work useful, please cite us at
```
@article{yan2025peplmgnn,
  title={PepLM-GNN: A Graph Neural Network Framework Leveraging Pre-trained Language Models for Peptide-Protein Binding Prediction},
  author={Ke Yan, Meijing Li, Shutao Chen, Tianyi Liu, and Bin Liu},
  journal={submitted},
  year={2025},
  publisher={}
}

```
