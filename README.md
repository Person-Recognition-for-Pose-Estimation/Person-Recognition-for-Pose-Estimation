This repo is the code for for *Using Facial Recognition for Selective Pose Detection* 
by William Parker for the Master of Science in Artificial Intelligence 
program at San José State University.

python training/round_robin_trainer.py --logging wandb

## Instructions:

All work has been done on Ubuntu 22.04 LTS and Python 3.10.12. All other configurations are untested.

1. Create a venv: `python3 -m venv venv`
2. Activate the venv: `source venv/bin/activate`
3. Install all the requirements: `pip install -r requirements.txt`
4. Download the component models `python scripts/download_models.py`
5. Combine the models into a multi-task model: `python scripts/modify_models.py`
6. Download the datasets by following the instructions in: `download_datasets.ipynb`
7. Train the model: `python training_ultra/test_yolo.py`

## Project Organization:
- `component_models/`: Holds the original downloads of the separate models. Created by `download_models.py`
- `dataset_folders/`: Holds the datasets needed to be stored locally. 
- `edited_components/`: Holds the modified versions of the component models and the untrained combined model. Created by `modify_models.py`
- `libs/`: Holds individual files from AdaFace that are needed for the project.
- `notebooks/`: Holds the notebooks used in development to create the model.
  - `download_models.ipynb`: Downloads the models from the HuggingFace hub.
  - `modify_models.ipynb`: Modifies the models and combines into the multi-task model.
  - `train_model.ipynb`: Trains the model.
- `requirements/`: Holds all the requirements for all of the projects for each component model.
- `scripts/`: Same as `notebooks/` but in script form for easy execution.
- `download_datasets.ipynb`: Installation guide for the datasets.
- `requirements.txt`: All the requirements for the project.