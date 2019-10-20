# Clinical XLNet

We pretrain the [XLNet-Base model](https://github.com/zihangdai/xlnet) on the [MIMIC-III](https://mimic.physionet.org/about/mimic/) discharge summary dataset for 200K steps with batch size 8 on top of the XLNet-Base checkpoint and the loss dropped from 1.90 to 0.63. 

Tensorflow checkpoint and Pytorch checkpoint are available:

[TF Model](https://hu-my.sharepoint.com/:u:/g/personal/kexinhuang_hsph_harvard_edu/ETPGuo3jePVGpd0twqpcGx4BslnQ3znSF-UVrWoG2rPhcQ?e=FeKAZW)

[Torch Model](https://hu-my.sharepoint.com/:u:/g/personal/kexinhuang_hsph_harvard_edu/ERPrsMddJMhDoltA5EniFk8BkLKoRF3O5JmYfCThvG3ctQ?e=tbiotE)

Downstream evaluations on specific clinical task will be out in a few months.

Example usage also coming soon...

Cite the [ClinicalBERT](https://github.com/kexinhuang12345/clinicalBERT) paper for now:
```
@article{clinicalbert,
author = {Kexin Huang and Jaan Altosaar and Rajesh Ranganath},
title = {ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission},
year = {2019},
journal = {arXiv:1904.05342},
}

```
