import matplotlib.pyplot as plt
import numpy as np
import random
import sys

def get_line_plot(x,func):
    '''
        Obtém os coeficientes angular e linear da reta da função
        no formato y(x) = mx + b

        Args:
            x: valores do eixo x
            func: vetor de pesos [w0,w1,w2] correspondente
        Result:
            valores correspondentes aos parâmetros func e x passados como parâmetro
    '''

    # Teste para evitar divisão por zero
    if np.all(func==0):
        return np.zeros(len(x))

    m = -func[0]/func[1]
    b = -func[2]/func[1]
    
    # Gera a linha dados os pontos do eixo x
    return x * m + b

def plot(X,y_true,t_func,h_func):
    '''
        Função que "plota" os dados de treinamento junto à hipótese

        Args:
            X: valores dos dados de treinamento [-1,1] x [-1,1]
            Y: labels dos dados de treinamento 
            h_func: função hipótese do perceptron

    '''

    plt.figure()
    
    # Filtra e plota os exemplos negativos
    x_negs, y_negs = [X[i][0] for i in range(len(X)) if y_true[i] < 0], [X[i][1] for i in range(len(X)) if y_true[i] < 0]
    plt.scatter(x_negs,y_negs,color='red')
    
    # Filtra e plota os exemplos positivos
    x_pos, y_pos = [X[i][0] for i in range(len(X)) if y_true[i] > 0], [X[i][1] for i in range(len(X)) if y_true[i] > 0]
    plt.scatter(x_pos,y_pos,color='green')
    
    
    # Plota target function
    xplt = np.linspace(-1, 1, 100)
    yplt = get_line_plot(xplt, t_func)
    
    plt.plot(xplt,yplt, color='blue',label='$f(x)$')
    
    # Plota a hipótese
    yplt2 = get_line_plot(xplt, h_func)
    plt.plot(xplt,yplt2,color='black',label='$g(x)$')

    plt.legend()
    plt.grid(True)
    
    plt.show()

def create_target_function():
    '''
        Gera a função target, desconhecida, que determina a classificação dos dados de treinamento

        Result:
            array com três posições w0 = 0, w1 = m, w2 = b, onde m e b são os coeficientes angular e linear, respectivamente
    '''

    #Pego dois pontos aleatórios no espaço [-1,1] x [-1,1]
    p0 = random.uniform(-1,1), random.uniform(-1,1)
    p1 = random.uniform(-1,1), random.uniform(-1,1)

    #Calcula o coeficiente angular 
    m = (p1[1]-p0[1])/(p1[0]-p0[0])
    
    # Calcula o coeficiente linear
    b = p0[1] - (p0[0] * m)
    
    # Gero a função target 
    # w0 = m, w1 = -1, w2= b
    # y = m*x + b -> m*x -y + b = 0
    return np.array([m,-1,b])

def generate_training_data(N,f_target):
    '''
        Gera um conjunto de dados linearmente separável pela função target (desconhecida pelo perceptron)

        Args:
            N: número de exemplos a serem gerados
            f_target: função target que separa esses dados
        Result:
            X_sample: vetor contendo os pontos (x,y) dos dados de treinamento e;
            y_sample: vetor contendo as labels dos dados de treinamento
    '''
    y_sample = []

    # Gerando a amostra de N pontos no espaço [-1,1] x [-1,1]
    X_sample = np.random.uniform(low=-1, high=1, size=(N,2))
    X_sample = [np.concatenate((X_sample[i],np.array([1.]))) for i in range(N)]
    for i in range(N):
        y_sample.append(1 if np.dot(f_target,X_sample[i]) > 0 else -1) # Classificação do ponto gerado segundo a função target passada como parâmetro
    
    return X_sample, np.array(y_sample)

class Perceptron:
    '''
        Classe que implementa o Perceptron e o PLA
    '''

    def __init__(self,show_plot=False):
        self.g = np.zeros(3)
        self.show_plot = show_plot

    def __sign(self,func,pt):
        '''
            Define a classificação do ponto de acordo com o produto interno entre o ponto e a função
            Args:
                func: função para classificar o dado em +1 ou -1
                pt: ponto no plano a ser classificado
            Result:
                valor inteiro correspondente ao rótulo do dado
        '''
        return 1 if np.dot(func,pt) > 0 else -1

    def __update_weights(self,X,y):
        '''
            Atualiza o vetor de pesos da função hipótese g
            O vetor de pesos é atualizado de acordo com um dos pontos classificados incorretamente
            O ponto é escolhido de forma aleatória dentro do conjunto

            Args:
                X: dados de treinamento classificados de forma incorreta
                y: classificação dada por g (incorreta)
            Result:
                os pesos da reta g atualizados
        '''

        #Escolhe um ponto aleatoriamente
        i = random.randint(0,len(X)-1)

        # weight vector
        w = self.g
        
        #Atualiza os pesos
        # w(t+1) = w(t) + y(t)*x(t)
        # Nesse caso, vamos usar w(t+1) = w(t) - y(t)*x(t) porque consideramos m*x - y + b = 0
        w = self.g - y[i]*X[i]
        
        return w

    def __get_miss_classified_examples(self,X,y_pred,y_true):
        '''
            Compara os rótulos de cada exemplo y_pred com y_true para retorna quais pontos não foram classificados corretamente por g

            Args:
                X: pontos (x,y) do conjunto de treinamento
                y_pred: rótulos previstos por g (hipótese)
                y_true: valores reais do rótulos de acordo com a função target desconhecida
        '''
        X_mislabeled,y_mislabeled = [],[]
        for i in range(len(y_true)):
            if y_pred[i] != y_true[i]:
                X_mislabeled.append(X[i])
                y_mislabeled.append(y_pred[i])
        return X_mislabeled,y_mislabeled

    def predict(self, X, function):
        '''
            Realiza a predição de todos os pontos do conjunto de treinamento de acordo com a função passada como parâmetro

            Args:
                X: pontos (x,y) do conjunto de treinamento
                function: função que tenta separar os pontos
            Return:
                lista de rótulos de classificação de acordo com a função function
        '''
        return [self.__sign(function,x) for x in X]

    def train(self,X,y,f):
        '''
            Perceptron Learning Algorithm

            Começando com g contendo apenas valores nulos, a reta é atualizada de acordo com os exemplos que não foram classificados corretamente
            O algoritmo termina quando não há mais pontos classificados incorretamente.
        '''

        # Inicializo o número de iterações como zero
        n_iteractions = 0

        while True:

            # Inicializo os arrays de predição e exemplos que não foram classificados corretamente
            X_miss_classified, y_miss_classified, predicted = [], [], []


            # Predição dos exemplos pela função hipótese
            predicted = self.predict(X,self.g)
        
            # Coleto todos os pontos que não foram classificados corretamente
            X_miss_classified,y_miss_classified = self.__get_miss_classified_examples(X,predicted,y)
        
            print(f'Número de pontos classificados de forma errada: {len(X_miss_classified)}')

            # Desenho os pontos e as funções na tela (opcional)
            if self.show_plot:
                plot(X,y,f,self.g)

            # Se todos os pontos foram classificados corretamente, encerra o programa
            if len(X_miss_classified) == 0:
                break

            # Uso os pontos que não foram classificados corretamente para ajustar os pesos de g
            self.g = self.__update_weights(X_miss_classified,y_miss_classified)
            
            # Cada vez que atualizo os pesos, conto uma iteração
            n_iteractions += 1
            
        return n_iteractions


def main():

    # Quantidade de pontos e interações
    N = 10
    total_iteractions = 0

    # True para mostrar o gráfico a cada iteração; False cc
    show_plot = True

    for i in range(1):

        f_target = create_target_function()
        
        # Inicializa a função objetivo e gera o conjunto de treinamento
        X,y = generate_training_data(N,f_target)

        # Inicializa a classe perceptron
        model = Perceptron(show_plot=True)

        print(f'Rodada {i+1}\n')
        n_iteractions = model.train(X,y,f_target)
        print(f'Fim da rodada {i+1}. Número de iterações: {n_iteractions}\n')
        total_iteractions += n_iteractions
        
    print(total_iteractions/1000)

if __name__ == '__main__':
    sys.exit(main())