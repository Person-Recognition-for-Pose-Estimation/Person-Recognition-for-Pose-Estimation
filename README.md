## Instructions

All work has been done on Ubuntu 22.04 LTS and Python 3.10.12. All other configurations are untested.

1. Create a venv: `python3 -m venv venv`
2. Activate the venv: `source venv/bin/activate`
3. Install all the requirements: `pip install -r requirements.txt`
4. Download the component models `download_models.ipynb`
5. Combine the models into a multi-task model: `modify_models.ipynb`
6. Download the datasets: `download_datasets.ipynb`
7. Train the model: `train_model.ipynb`