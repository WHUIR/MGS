# MGS

## Code
This is the official implementation of [*An Attribute-Driven Mirror Graph Network for Session-based Recommendation*](https://doi.org/10.1145/3477495.3531935) from SIGIR 2022.

## Environment
* Python=3.7
* PyTorch=1.10.1
* numpy=1.20.3

## Usage
### Datasets

Our data has been preprocessed and is available at https://www.dropbox.com/sh/h548vcds8a4m3qs/AABH-YavkkoNFPR_RMTtmILOa?dl=0. You need to download the *datasets* folder and put it under the root. All the original dataset files are available at https://www.dropbox.com/sh/dbzmtq4zhzbj5o9/AAAMMlmNKL-wAAYK8QWyL9MEa/Datasets?dl=0&subfolder_nav_tracking=1.

* Diginetica dataset has a category attribute. Its attribute information can be found at https://competitions.codalab.org/competitions/11161#learn_the_details-data2.
* 30music dataset has an artist attribute. Each item's attribute value can be found in the original dataset file.
* The brief version of the Tmall dataset is also available on that website. And you can find its corresponding original file containing attribute information at https://tianchi.aliyun.com/dataset/dataDetail?dataId=42. It includes a category and a brand attribute.

### Train and test

Train and evaluate the model:

```sh
python main.py --dataset Tmall
```

The sequence length of Diginetica dataset is relatively long. You can also train the model by utilizing 2 GPUs:

```
python main_dual_GPU.py --dataset diginetica
```

## Citation

```
@inproceedings{DBLP:conf/sigir/LaiMZLWS22,
  author    = {Siqi Lai and
               Erli Meng and
               Fan Zhang and
               Chenliang Li and
               Bin Wang and
               Aixin Sun},
  editor    = {Enrique Amig{\'{o}} and
               Pablo Castells and
               Julio Gonzalo and
               Ben Carterette and
               J. Shane Culpepper and
               Gabriella Kazai},
  title     = {An Attribute-Driven Mirror Graph Network for Session-based Recommendation},
  booktitle = {{SIGIR} '22: The 45th International {ACM} {SIGIR} Conference on Research
               and Development in Information Retrieval, Madrid, Spain, July 11 -
               15, 2022},
  pages     = {1674--1683},
  publisher = {{ACM}},
  year      = {2022},
  url       = {https://doi.org/10.1145/3477495.3531935},
  doi       = {10.1145/3477495.3531935},
  timestamp = {Fri, 08 Jul 2022 17:25:07 +0200},
  biburl    = {https://dblp.org/rec/conf/sigir/LaiMZLWS22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
