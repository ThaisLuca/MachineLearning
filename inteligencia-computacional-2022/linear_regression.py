
import numpy as np
import random
import sys


class LinearRegression:
    '''
        Classe que implementa o algoritmo de Regressão Linear
    '''
    
    def __init__(self):
        self.w = np.zeros(3)

        
    def fit(self,X,y):
        '''
            Função que calcula os pesos da Regressão Linear
            a partir dos dados de treinamento
            
            w = X'*y
            
            onde X' é a pseudo-inversa de X, dada por:
            X' = (X.T*X)^(-1) * X.T
            
            Este é chamado de one-step learning porque
            tem apenas um passo de aprendizado. 
            Basta resolver o sistema acima para obter os
            valores de w que resolve o sistema Xw = y

            Args:
                X: valores dos dados de treinamento [-1,1] x [-1,1]
                y: labels dos dados de treinamento 
        '''
        
        # Para encontrar a reta que melhor descreve os pontos, 
        # queremos resolver o seguinte sistema
        # X.T * X * w = X.T * y
        
        # Primeiro multiplicamos X e y por suas matrizes transpostas
        # aqui é usado um operador de matrizes
        A = np.matmul(X.T,X)
        b = np.matmul(X.T,y)
        
        # Para encontrar o vetor w, precisamos resolver o sistema Ax=b
        # vamos usar a seguinte função do Numpy
        self.w = np.linalg.solve(A, b)
    
    def __sign(self,pt):
        '''
            Define a classificação do ponto de acordo com o produto interno 
            entre o ponto e o vetor w
            
            Args:
                pt: ponto a ser classificado
            Result:
                valor inteiro correspondente ao rótulo do dado segundo w
        '''
        return 1 if np.dot(self.w,pt) > 0 else -1
    
    def predict(self, X):
        '''
            Realiza a predição de todos os pontos do conjunto de treinamento 
            de acordo com o vetor w

            Args:
                X: pontos (x,y) do conjunto de treinamento
            Return:
                lista de rótulos de classificação de acordo com w
        '''
        return [self.__sign(x) for x in X]

def get_error(y,preds):
    '''
        Calcula o erro binário de todas as predições
        Na prática, conta quantas vezes o valor de saída
        foi diferente do valor esperado

        Args:
            X: pontos (x,y) do conjunto de treinamento
        Return:
            erro médio binário
    '''
        
    # Neste problema, temos uma saída binária,
    # por isso vamos calcular quantas vezes 
    # h(x) é diferente de f(x)
    # onde h é a hipótese e f a target
    error = 0
    for y_true,y_pred in zip(y,preds):
        if y_true != y_pred:
            error += 1/len(preds)
    return error

def main():
    # Quantidade de pontos e o valor de E_in (erro dentro da amostra)
    N = 100
    E_in = 0

    for i in range(1000):

        # Inicializa a função objetivo
        f_target = create_target_function()

        # Gera o conjunto de treinamento
        X,y = generate_training_data(N,f_target)

        # Inicializa a classe que implementa a Regressão Linear
        model = LinearRegression()

        # Treinamento do modelo - one step learning
        model.fit(X,y)
        
        # Uso o modelo pra fazer a predição dos dados dentro da amostra segundo a hipótese
        preds = []
        preds = model.predict(X)
        
        # Calculo o erro da função hipótese encontrado pelo algoritmo
        E_in += get_error(y,preds)

    print(E_in/1000)

if __name__ == '__main__':
    sys.exit(main())