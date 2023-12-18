# Reproduction of "A Simple Framework for Contrastive Learning of Visual Representations"

## ECBM E4040 - Deep Learning and Neural Networks, Columbia University, Fall 2023

### Introduction
This repository contains our team's project for ECBM E4040 at Columbia University. Our work focuses on reproducing the findings of the paper "A Simple Framework for Contrastive Learning of Visual Representations" by Ting Chen, Simon Kornblith, et al.

### Team Members
- Yilin Ye: [yy3152, yy3152@columbia.edu]
- Zhenyi Yang: [zy2540, zy2540@columbia.edu]
- Haoyu He: [hh2982, hh2982@columbia.edu]

### Built With
- Tensorflow 2
- Keras
- Google Cloud Platform

### Our project involves:

1. Coding Replication of the Paper's Neural Network Model Using TensorFlow 2 and Python.
2. The Final Report That Summarizes the Original Paper, Provides a Brief Review of the Relevant Literature, and Presents the Details of Our Project Work and Results.

### Dataset

The dataset we are using is CIFAR-100, which is located in the tensorflow.keras.datasets module.

### Repository Structure
- '/augmentation': Contains all python code used for augmentation.
- '/saved_models': Contains our saved model.
- 'config.yaml': Contains our configs used to train and test the model.
- 'train.ipynb': Our main Jupyter notebook.
- 'model.py': Our built model using transfer learning.

### Description of key functions of each file in the project

- 'augment_image'in 'train.ipynb': Apply a series of augmentations for SimCLR suitable for CIFAR-100.
- 'preprocess_for_simclr' in 'train.ipynb': Preprocesses and applies augmentation for SimCLR.
- 'contrastive_loss' in 'train.ipynb': Define contrastive loss function (NT-Xent loss)
- 'ResNetSimCLR' in 'model.py: Our Model.

###  Instructions on how to run the code

1. **Clone the Repository** 📂

   Use the following command to clone the repository:

   ```bash
   git clone https://github.com/d29f/E4040-SimCLR.git

2. **Install dependencies and activate virtual environment** 🔨

    Make sure you have virtualenv installed. If not, install it using pip:

    ```bash
    pip install virtualenv
    ```

    For macOS and Linux:
    ```bash
    cd E4040-SimCLR  
    virtualenv [venv]  # Replace [venv] with your virtual environment name
    source venv/bin/activate
    pip install -r requirements.txt
    ```

    For Windows:
    ```bash
    cd E4040-SimCLR  
    virtualenv [venv]  # Replace [venv] with your virtual environment name
    venv\Scripts\activate
    pip install -r requirements.txt
    ```
    
4. **Run the Jupyter notebook** 🚀
    Start the server using the following command:
    ```bash
    jupyter notebook train.ipynb 
    ```

### Contribution
