# TReNDS Neuroimaging
The first place solution to the Kaggle "TReNDS Neuroimaging" competition.

By team [Nikita Churkin](https://www.kaggle.com/churkinnikita) and [Dmitry Simakov](https://www.kaggle.com/simakov)

## Data

We will be using [original dataset](https://www.kaggle.com/c/trends-assessment-prediction/data). Download these from Kaggle and put all in a `data/raw` folder. Your directory structure should look as follows:
```
.
├── age.py
├── agg_feats.py
├── all_labels.py
├── compute_biases.py
├── create_img_statistics.py
├── create_pca_dl_feats.py
├── create_submission.py
├── d11.py
├── d12.py
├── d21.py
├── d22.py
├── data
│   └── raw
│       ├── fMRI_test
│       ├── fMRI_train
│       ├── fnc.csv
│       ├── loading.csv
│       ├── reveal_ID_site2.csv
│       ├── sample_submission.csv
│       └── train_scores.csv
├── resave_imgs.py
├── site_classifier.py
├── trends.py
├── tils.py
├── LICENSE
├── README.md
├── model_summary.pdf
└── requirements.txt

```

## Hardware
This code was tested in the following setting:
* OS: Ubuntu 18.04
* RAM: 64GB
* CPU: 12 cores (24 threads)


## Additional disk space requirements:
* 450gb for resaved fMRI data
* ~30gb for PCA/DL models
* all saved models (7 seeds for 5 targets) ~160gb of disk space


## Execution time:
* 13 hours for PCA features creation
* 47 hours for DL features creation
* ~1 hour for other features
* ~2.5 hours for one seed model training and inference (final submission is blend for 7 validation seeds)


## Requirements
We provide a `requirements.txt` file to install the dependencies through pip. 

## Submission

To reproduce our full solution one should run the following scripts:
```
python resave_imgs.py
python create_img_statistics.py
python create_pca_dl_feats.py
python site_classifier.py
python agg_feats.py 
python compute_biases.py
python all_labels.py
python create_submission.py
```

## Scripts

* [1. resave_imgs](resave_imgs.py)

We resaved original 3D fMRI data in pickle format with right channel order and in float32. It is needed for faster data loading in `create_pca_dl_feats` script.

* [2. create_img_statistics](create_img_statistics.py)

We calculated simple statistics (mean, std, quantiles) for each of 53 feature channels of original 3D fMRI `.mat` files.

* [3. create_pca_dl_feats](create_pca_dl_feats.py)

The longest script. Its execution took ~2.5 days in the following hardware. We used Incremental PCA on fMRI data with n_components 200, batch-size 200. Channels were splitted in groups by 10 and flattened inside them (6 groups in total). As a result, we got 1200 PCA features. Dictionary-learning (DL) params: n-components 100, batch-size 100 and n-iters 10 (the same scheme with channels splitting). There were 600 dl features.

* [4. site_classifier](site_classifier.py)

This script trained and inferensed site2 classifier. We applied StandartScaler for train + test data. Regression ElasticNet was used for modeling. Our model detected ~1400 new site2 observations.

* [5. agg_feats](agg_feats.py)

Simple statistics for different fnc groups were calculated here.

* [6. compute_biases](compute_biases.py)

Offsets for test set were calculated to minimize differences in train-test distributions.

* [7. all_labels](all_labels.py)

We trained our ensemble and predicted test data. More datailes can be found in our technical report.

* [8. create_submission](create_submission.py)

We blended all validation seeds and applied postprocessing. Final submission can be found in `predicts/submission/`

## References & Pointers

* [The kaggle competition](https://www.kaggle.com/c/trends-assessment-prediction)

* [A detailed blog post on our solution](https://www.kaggle.com/c/trends-assessment-prediction/discussion/163017)


