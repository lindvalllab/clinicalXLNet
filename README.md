# Clinical XLNet

We pretrain the [XLNet-Base model](https://github.com/zihangdai/xlnet) on the [MIMIC-III](https://mimic.physionet.org/about/mimic/) discharge summary dataset for 200K steps on top of the XLNet-Base checkpoint and achieved loss xxx. 

Tensorflow checkpoint and Pytorch checkpoint are available:

[TF Model]()

[Torch Model]()

Downstream evaluations on specific clinical task will be out in a few months.

Cite the [ClinicalBERT](https://github.com/kexinhuang12345/clinicalBERT) paper for now:
```
@article{clinicalbert,
author = {Kexin Huang and Jaan Altosaar and Rajesh Ranganath},
title = {ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission},
year = {2019},
journal = {arXiv:1904.05342},
}

```
