import os
import requests
import pathlib

# HF folder
hf_folder = "components"

# Models to download
person_detector = "yolo11n.pt"
person_detector_path = os.path.join(hf_folder, person_detector)
face_detector = "yolov11n-face.pt"
face_detector_path = os.path.join(hf_folder, face_detector)
face_identifier = "adaface_ir50_ms1mv2.ckpt"
face_identifier_path = os.path.join(hf_folder, face_identifier)

repo_id = 'Jaspann/Person-Recognition-for-Pose-Estimation'

filepath = pathlib.Path(__file__).parent.resolve()
component_models_dir = os.path.join(filepath, "..", "component_models")

# Create output directory if it doesn't exist
if not os.path.exists(component_models_dir):
    os.makedirs(component_models_dir)


def hf_download(model_name: str, model_path: str, repo_id: str, folder_path: str) -> None:
    """
    Download a model from the Hugging Face model hub.

    Args:
        model_name (str): The name of the model file to download.
        model_path (str): The path where the model file will be saved.
        repo_id (str): The repository ID on Hugging Face.
        folder_path (str): The local folder path where the model will be saved.
    """

    if os.path.exists(os.path.join(folder_path, model_name)):
        print(f"{model_name} already exists in {folder_path}")
        return

    url = f"https://huggingface.co/{repo_id}/resolve/main/{model_path}?download=true"
    response = requests.get(url)

    if response.status_code == 200:
        with open(os.path.join(folder_path, model_name), "wb") as f:
            f.write(response.content)
    else:
        print(f"Error: {response.status_code}")

# Run the downloads
hf_download(person_detector, person_detector_path, repo_id, component_models_dir)
hf_download(face_detector, face_detector_path, repo_id, component_models_dir)
hf_download(face_identifier, face_identifier_path, repo_id, component_models_dir)