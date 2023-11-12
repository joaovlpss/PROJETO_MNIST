import os
import tkinter as tk 
from tkinter import filedialog 
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt  

# Função para carregar a matriz de confusão a partir de um arquivo
def load_conf_matrix(model_path):
    conf_matrix = np.load(model_path)
    return conf_matrix

# Função para calcular as acurácias total e por label
def calculate_accuracies(conf_matrix):
    total_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)  # Calcula a acurácia total
    label_accuracies = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)  # Calcula a acurácia por label
    return total_accuracy, label_accuracies

# Função para visualizar a matriz de confusão e as acurácias
def visualize_conf_matrix(conf_matrix, total_accuracy, label_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Cria uma figura com dois subplots

    # Plot da matriz de confusão no primeiro subplot
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Matriz de confusão')  # Define o título do subplot
    ax1.set_ylabel('Label Verdadeira')  # Define o rótulo do eixo Y
    ax1.set_xlabel('Label Predita')  # Define o rótulo do eixo X

    # Prepara o texto de acurácia para o segundo subplot
    accuracy_text = f"Acurácia Total: {total_accuracy:.2f}\n\n"
    accuracy_text += "\n".join([f"Label {idx}: {acc:.2f}" for idx, acc in enumerate(label_accuracies)])

    # Exibe o texto de acurácia no segundo subplot
    ax2.text(0.5, 0.5, accuracy_text, ha='center', va='center', fontsize=12)
    ax2.axis('off')  # Desativa os eixos

    plt.tight_layout()  # Ajusta o layout para evitar sobreposição
    plt.show()  # Exibe o gráfico

# Função principal
def main():
    root = tk.Tk()  # Cria a janela principal da GUI
    root.title('Visualização do Classificador Naive Bayes')  # Define o título da janela

    # Função chamada quando o botão 'Open Model' é clicado
    def on_open():
        # Abre a caixa de diálogo para escolher um modelo
        file_path = filedialog.askopenfilename(initialdir='Projeto_MNIST/saved_model', title='Selecionar Modelo', filetypes=[('Arquivos Numpy', '*.npy')])
        if file_path:
            conf_matrix = load_conf_matrix(file_path)  # Carrega a matriz de confusão
            total_accuracy, label_accuracies = calculate_accuracies(conf_matrix)  # Calcula as acurácias

            # Visualiza a matriz de confusão e as acurácias
            visualize_conf_matrix(conf_matrix, total_accuracy, label_accuracies)

    # Botão para abrir a caixa de diálogo
    open_button = tk.Button(root, text='Abrir modelo', command=on_open)
    open_button.pack()  # Adiciona o botão à janela

    root.mainloop()  # Inicia o loop principal da GUI

# Executa a função main se este script for o principal
if __name__ == "__main__":
    main()
