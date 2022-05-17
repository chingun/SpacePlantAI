# SPACE PLANT

The VIT + EfficientNet state of the art Neural Network architecture for the next BILLION dollar company.

### Requirements

The `train.py` is runnable simply with `python train.py`. However, you are responsible for downloading the main dataset into the folder directly outside your <b>SpacePlantAI</b> directory. If you have installed anaconda, you can run the following command :

```conda env create -f spaceplant_env.yml```

### Training a model

In order to train a model on the SpacePlant dataset, run the following command:

```python vit_efficientnet.py --lr=0.05 --n_epochs=80 --k 1 --model=vit --root=path_to_data --save_name_xp=VisionTransformer```
```python vit_efficientnet.py --lr=0.05 --n_epochs=80 --k 1 --model=efficientnet --root=path_to_data --save_name_xp=xp1```

You must provide in the "root" option the path to the train val and test folders. 

The "save_name_xp" option is the name of the directory where the weights of the model and the results (metrics) will be stored.

You can check out the different options in the file cli.py.

├── cli.py
├── cuda_11.0.2_450.51.05_linux.run
├── dense_resnet_main.py
├── epoch.py
├── LICENSE
├── README.md
├── results
│   ├── xp1
│   │   ├── xp1_weights_best_acc.tar
│   │   └── xp1_weights.tar
│   └── xp_earned
│       ├── xp_earned_weights_best_acc.tar
│       └── xp_earned_weights.tar
├── spaceplant_env.yml
├── utils.py
└── vit_efficientnet.py



© ℗ ® Access reserved for Space Plant subsidiaries and employees only.

