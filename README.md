# MNIST Project - EN

## Overview
The MNIST Project is designed to demonstrate the capabilities of machine learning in recognizing handwritten digits. It uses the MNIST dataset to train a Naive Bayes classifier and provides tools for data visualization.

Datasets were not added to this repo due to size, but can be easily downloaded and added into 'data/' as 'mnist_train.csv' and 'mnist_test.csv' from 'https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data'.

## Project Structure
- `data/`: Contains the training and testing data.
  - `mnist_train.csv`
  - `mnist_test.csv`
- `mnist_images/`: Stores variance and average images for each label.
- `model/`: Contains the script used for training the model.
  - `model_trainer.py`
- `saved_model/`: Stores trained models and their corresponding confusion matrices.
  - `naive_bayes_X_Y_conf_matrix.npy`
  - `naive_bayes_X_Y.joblib`
  - `naive_bayes_all_conf_matrix.npy`
  - `naive_bayes_all.joblib`
- `visualization/`: Contains visualization tools for the project.
  - `model_visualization.py`
  - `generate_images.py`

## Key Components
### model_trainer.py
This script trains a Gaussian Naive Bayes model using the MNIST dataset. It generates models for each digit pair (0-9) and a combined model for all digits. The trained models and their confusion matrices are saved in the `saved_model/` directory.

### model_visualization.py
A GUI tool for visualizing the confusion matrices and accuracies of the trained models. It allows users to select a specific model and displays its performance metrics.

### generate_images.py
Generates and saves 28x28 images representing the variance and average of each digit label in the MNIST dataset. The images are saved in the `mnist_images/` directory.

## Usage
To use the project, run the `model_trainer.py` script to train the models (This step comes pre-made). Then, use `model_visualization.py` to view the performance of these models, by navigating to the `saved_model` folder and choosing a model to visualize. You can generate variance and average images for each label by running `generate_images.py` (also pre-made).


# Projeto MNIST - PT-BR

## Visão Geral
O Projeto MNIST é projetado para demonstrar as capacidades do aprendizado de máquina no reconhecimento de dígitos escritos à mão. Ele utiliza o conjunto de dados MNIST para treinar um classificador Naive Bayes e fornece ferramentas para visualização de dados.

O conjunto de dados não foi adicionado a este repositório GitHub devido ao tamanho excessivo dos dados, mas o dataset pode ser baixado facilmente pelo link 'https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data' e adicionado a 'data/'.

## Estrutura do Projeto
- `data/`: Contém os dados de treino e teste.
  - `mnist_train.csv`
  - `mnist_test.csv`
- `mnist_images/`: Armazena imagens de variância e média para cada label.
- `model/`: Contém o script usado para treinar o modelo.
  - `model_trainer.py`
- `saved_model/`: Armazena os modelos treinados e suas respectivas matrizes de confusão.
  - `naive_bayes_X_Y_conf_matrix.npy`
  - `naive_bayes_X_Y.joblib`
  - `naive_bayes_all_conf_matrix.npy`
  - `naive_bayes_all.joblib`
- `visualization/`: Contém ferramentas de visualização para o projeto.
  - `model_visualization.py`
  - `generate_images.py`

## Componentes Principais
### model_trainer.py
Este script treina um modelo Gaussian Naive Bayes usando o conjunto de dados MNIST. Ele gera modelos para cada par de dígitos (0-9) e um modelo combinado para todos os dígitos. Os modelos treinados e suas matrizes de confusão são salvos no diretório `saved_model/`.

### model_visualization.py
Uma ferramenta GUI para visualizar as matrizes de confusão e acurácias dos modelos treinados. Permite que os usuários selecionem um modelo específico e exibam suas métricas de desempenho.

### generate_images.py
Gera e salva imagens de 28x28 representando a variância e a média de cada label de dígito no conjunto de dados MNIST. As imagens são salvas no diretório `mnist_images/`.

## Uso
Para usar o projeto, execute o script `model_trainer.py` para treinar os modelos (esse passo já está feito). Em seguida, use `model_visualization.py` para visualizar o desempenho desses modelos, navegando até a pasta `saved_model` e escolhendo o modelo desejado. Você pode gerar imagens de variância e média para cada label executando `generate_images.py` (passo também pré-realizado).
