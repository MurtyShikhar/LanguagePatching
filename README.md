## Setup

Install all dependencies using `conda`:

```
conda env create -f environment.yml
conda activate lang-patching
pip install -e .
```

## Training Pipeline
Note that this repository uses `hydra` for managing hyperparameters and experiments. The configs we use for training can be found under `config`. `hydra` creates unique outputs for every experiment under the directory `output`. 

### Creating Patch Finetuning Data
To start, create synthetic data for patch finetuning using the yaml format (some examples are in the `PATCH_DIR` folder), and then use `convert_yaml_to_data.py` to create json files. The JSON files used in our experiments can be found in the `PATCH_DIR` folder. 

### Training Patchable Models

The entry script for training patchable models is `train_models.py`. Run it as:

``` python train_models.py train.save_path={SAVE_PATH} +protocol={protocol} +patch_type={SUB_FOLDER} +multitask_sst=True +train.load_path={TASK_FINETUNED_MODEL} +learnt_interpreter={True/False}```

- {SAVE_PATH}: path where the patchable model will be saved
- {protocol}: can be one of
  - `simple`: If you want to train a model on just the task ("Task Finetuning")
  - `patch_finetuning_conds`: train a patchable model for Sentiment Classification
  - `patch_re`: to train a patchable model for Relation Extraction
- {SUB_FOLDER}: one of the folders in the PATCH_DIR directory. To train models with override patches, use `override_patch_data` and to train a model with feature based patches, use `feature_based_patch_data`.
- learnt_interpreter: set this to `True` to train feature based patches.



### Model Checkpoints
Checkpoints for models used in this work can be found at this [link](https://drive.google.com/drive/folders/1TWdPW7QS6um21cDlBzH26-gs3fkDnoat?usp=share_link). We also provide notebooks to reproduce various Tables in the paper. To reproduce results:
- For Table-2, see the notebooks with `checklist` in the name
- For Table-3,4,5 please follow the instructions in the notebook `override_patches_sentiment.ipynb` and `orig_model_results.ipynb`
- For Figure-4, use `finetuning_experiments.py`


To cite this paper, use:
```
@inproceedings{murty2022patches,
    title = "Fixing Model Bugs with Natural Language Patches",
    author = "Murty, Shikhar  and
      Manning, Christopher  and
      Lundberg, Scott  and
      Ribeiro, Marco Tulio",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022",
}
```
