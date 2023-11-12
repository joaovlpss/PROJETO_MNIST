import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import joblib
import os

def config_files():
    # Obtem o diretório deste script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Define o caminho do projeto em relação ao diretorio do script
    project_path = os.path.join(script_dir, '..')
    # Define o caminho para os dados de treinamento
    train_data_path = os.path.join(project_path, 'data/mnist_train.csv')
    # Define o caminho para os dados de teste
    test_data_path = os.path.join(project_path, 'data/mnist_test.csv')
    # Define o caminho para o modelo
    model_path = os.path.join(project_path, 'model')
    # Define o caminho para o modelo salvo
    saved_model_path = os.path.join(project_path, 'saved_model')
    # Retorna os caminhos
    return train_data_path, test_data_path, saved_model_path

def train_and_save_model(X_train, y_train, X_test, y_test, name, saved_model_path):
    # Cria um modelo GaussianNB
    model = GaussianNB()
    # Treina o modelo com os dados de treino
    model.fit(X_train, y_train)
    # Faz a previsão com os dados de teste
    y_pred = model.predict(X_test)
    # Cria a matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Salva o modelo treinado
    joblib.dump(model, os.path.join(saved_model_path, f'{name}.joblib'))
    # Salva a matriz de confusao
    np.save(os.path.join(saved_model_path, f'{name}_conf_matrix.npy'), conf_matrix)

    # Retorna a matriz de confusao
    return conf_matrix

def main():
    # Configura os caminhos dos arquivos
    train_data_path, test_data_path, saved_model_path = config_files()
    
    # Carrega os dados de treino e teste
    train_data = pd.read_csv(train_data_path, header=None)
    test_data = pd.read_csv(test_data_path, header=None)

    # Define os nomes das colunas como strings de 1 a 784
    feature_names = [str(i) for i in range(1, 785)]

    # Separa os dados em features (X) e target (y)
    X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
    X_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]

    # Treina o modelo usando todos os dados e salva o modelo treinado
    conf_matrix_all = train_and_save_model(X_train, y_train, X_test, y_test, 'naive_bayes_all', saved_model_path)
    print("Matriz de confusao para todas as labels:")
    print(conf_matrix_all)

    # Loop para treinar um modelo para cada par de labels
    for i in range(9):
        for j in range(i+1, 10):
            # Filtra os dados de treino para incluir apenas as labels i e j
            filter_indices = y_train.isin([i, j])
            X_train_pair = X_train[filter_indices]
            y_train_pair = y_train[filter_indices]

            # Filtra os dados de teste para incluir apenas as labels i e j
            filter_indices_test = y_test.isin([i, j])
            X_test_pair = X_test[filter_indices_test]
            y_test_pair = y_test[filter_indices_test]

            # Treina o modelo usando apenas os dados filtrados e salva o modelo treinado
            name = f'naive_bayes_{i}_{j}'
            conf_matrix_pair = train_and_save_model(X_train_pair, y_train_pair, X_test_pair, y_test_pair, name, saved_model_path)
            print(f"Matriz de confusao para as labels {i} e {j}:")
            print(conf_matrix_pair)

# Se este script for o principal, executa a função main
if __name__ == "__main__":
    main()