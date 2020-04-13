# PyTorch Tutorial
To get started with the PyTorch tutorial, create a new environment as specified in the `env.yml` file via
```shell
conda env create -f env.yml
conda activate pytorch_tutorial
```
Then, create a dataframe that holds your images, e.g.
```shell
python data_to_df.py --folders $PWD/data/apples $PWD/data/bananas
```