# Alcohol_classifier
This project implements an end-to-end machine-learning pipeline for classifying images of alcoholic beverages into three categories: beer, wine, and whiskey. The framework is designed with a modular structure with separating data handling, model definition, training, evaluation, visualization, and inference into independent components. To ensure reproducibility and ease of deployment, the entire pipeline is containerized using Docker. This allows the same training and evaluation procedures to be executed consistently across different machines without dependency conflicts, making the framework portable and production-ready.


**Data:**
The dataset is organized in a standard folder structure (data/raw/{beer, wine, whiskey}), where each subdirectory represents a class. Images vary in resolution and appearance, reflecting real-world diversity in bottle shapes, labels, lighting conditions, and backgrounds. During preprocessing, all images are resized to a fixed resolution of 224×224 pixels to ensure consistent batching. The dataset can be accessed through: https://www.kaggle.com/datasets/surajgajul20/image-dataset-beer-whisky-wine.

**Model:**
The used framework and models are [TorchVision](https://github.com/pytorch/vision) and ResNet with pretrained weights (as a starting point). There is multiple ResNet models, but a ResNet-18 model will be used as a starting point for its low computational cost. More advanced/newer ResNet models might be used later in the project e.g. ResNet-50.

These models are trained for general object detection so the output would have to be changed for the 3 possible outputs in this project (beer, wine and whiskey)

**Tools:**
The pipeline of tools and helpers from the pytorch ecosystem will consist of some popular standard tools aswell as some that can facilitate our image classification problem:

- Model & Vision essentials
    - torchvision (contains pre-trained architectures like ResNet)
- Framework utilities
    - pytorch lightning
    - hydra (to parametrize the whole project using config files)
- Data versioning & logging
    - DVC (data version control, to keep track of the larger data files)
    - wandb (weights and biases)
- Deployment & reproducabilty
    - docker

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                   # CI/CD Workflows (GitHub Actions)
├── api/                       # Deployment & UI
│   ├── api.py                 # FastAPI backend (Inference)
│   ├── frontend.py            # Streamlit frontend (User Interface)
│   └── export_onnx.py         # Model conversion script
├── configs/                   # Hydra configuration files
├── data/                      # Data storage (Tracked by DVC)
│   ├── raw/                   # Raw images (Beer, Wine, Whiskey)
│   └── processed/             # Preprocessed data tensors
├── dockerfiles/               # Containerization
│   ├── api.Dockerfile         # Backend container
│   └── frontend.Dockerfile    # UI container
├── models/                    # Model Registry
│   ├── model.onnx             # Production model (ONNX format)
│   └── best_model.pt          # PyTorch checkpoint
├── src/                       # Project Source Code
│   └── alcohol_classifier/    # Core logic (train, evaluate, data)
├── tests/                     # Unit & Integration tests
├── requirements.txt           # Main dependencies
├── tasks.py                   # Automation (Invoke tasks)
└── pyproject.toml             # Project metadata & build system
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
