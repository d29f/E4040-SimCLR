# Reproduction of "A Simple Framework for Contrastive Learning of Visual Representations"

## ECBM E4040 - Deep Learning and Neural Networks, Columbia University, Fall 2023

### Introduction
This repository contains our team's project for ECBM E4040 at Columbia University. Our work focuses on reproducing the findings of the paper "A Simple Framework for Contrastive Learning of Visual Representations" by Ting Chen, Simon Kornblith, et al.

### Team Members
- Yilin Ye: [yy3152, yy3152@columbia.edu]
- Zhenyi Yang: [zy2540, zy2540@columbia.edu]
- Haoyu He: [hh2982, hh2982@columbia.edu]

### Built With
- Tensorflow (Version 2.x)
- Google Cloud Platform

### Our project involves:

1. A comprehensive review of the original paper.
2. Implementation and replication of the paper's neural network model using TensorFlow and Python.
3. Documentation and comparison of our results with the original paper.

### Datasets

(WIP) the README file has to provide detailed information on where the datasets are, how they can be accessed, and how they are used.

### Repository Structure
- '/augmentation': Contains all python code used for augmentation.
- '/data': Contains our datasets used to train and test the model.
- '/xxx.ipynb': Our main Jupyter notebook.

### Description of key functions of each file in the project

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
    jupyter notebook xxx.ipynb 
    ```

### Contribution