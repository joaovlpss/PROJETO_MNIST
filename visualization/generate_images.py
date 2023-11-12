import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    # Carrega os dados do CSV
    data = pd.read_csv(file_path, header=None)
    # Separa as labels dos pixels
    labels, pixels = data.iloc[:, 0], data.iloc[:, 1:]
    return labels, pixels

def calculate_statistics(pixels, labels):
    statistics = {}
    for label in range(10):
        # Filtra os pixels para cada label
        label_pixels = pixels[labels == label]
        # Calcula a média e a variância
        avg = label_pixels.mean().values.reshape(28, 28)
        var = label_pixels.var().values.reshape(28, 28)
        statistics[label] = {'average': avg, 'variance': var}
    return statistics

def save_images(statistics, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for label, stats in statistics.items():
        # Salva a imagem de média
        plt.imsave(f'{folder_name}/label_{label}_average.png', stats['average'], cmap='gray')
        # Salva a imagem de variância
        plt.imsave(f'{folder_name}/label_{label}_variance.png', stats['variance'], cmap='gray')

def main():
    # Obtem o diretório deste script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Define o caminho do projeto em relação ao diretorio do script
    project_path = os.path.join(script_dir, '..')
    train_data_path = os.path.join(project_path, 'data/mnist_train.csv')
    test_data_path = os.path.join(project_path, 'data/mnist_test.csv')
    images_path = os.path.join(project_path, 'mnist_images')

    # Carrega os dados de treino e teste
    train_labels, train_pixels = load_data(train_data_path)
    test_labels, test_pixels = load_data(test_data_path)

    # Combina os dados de treino e teste
    combined_labels = pd.concat([train_labels, test_labels])
    combined_pixels = pd.concat([train_pixels, test_pixels])

    # Calcula a média e a variância para cada label
    statistics = calculate_statistics(combined_pixels, combined_labels)

    # Salva as imagens de média e variância
    save_images(statistics, images_path)

if __name__ == "__main__":
    main()
