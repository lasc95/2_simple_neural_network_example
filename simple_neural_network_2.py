import numpy as np

# definimos las funciones de activación. En esta ocasión usaremos ReLu para las capas ocultas y la sigmoide para la capa de salida
# con sus respectivas derivadas
def relu(x):
    return np.maximum(0, x)

def relu_derivate(x):
    return np.where(x <= 0, 0, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivate(x):
    return x * (1 - x)

# creamos los pesos con valores aleatorios
np.random.seed(1)

weight0 = 2 * np.random.random((3, 4)) - 7
weight1 = 2 * np.random.random((4, 4)) - 1
weight2 = 2 * np.random.random((4, 1)) - 1

# creamos los datos de entrada y de salida
input_data = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
output_data = np.array([[0, 1, 1, 0]]).T

n_iteractions = 10000

# creamos la red neuronal con 10,000 iteraciones
for iteration in range(n_iteractions):
    # creamos nuestras capas, primero la capa de entrada, 2 ocultas y al final la de salida
    layer0 = input_data
    layer1 = relu(np.dot(layer0, weight0))
    layer2 = relu(np.dot(layer1, weight1))
    layer3 = sigmoid(np.dot(layer2, weight2))

    # obtenemos el error de la capa de salida
    layer3_error = output_data - layer3
    layer3_delta = layer3_error * sigmoid_derivate(layer3)

    if (iteration % n_iteractions) == 0:
        print("Error:" + str(np.mean(np.abs(layer3_error))))

    # obtenermos el error de la capa 2
    layer2_error = layer3_delta.dot(weight2.T)
    layer2_delta = layer2_error * relu_derivate(layer2)

    # obtenemos el error de la capa 1
    layer1_error = layer2_delta.dot(weight1.T)
    layer1_delta = layer1_error * relu_derivate(layer1)


    # actualizamos los pesos correspondientes para mejorar el rendimiento
    weight2 += layer2.T.dot(layer3_delta)
    weight1 += layer1.T.dot(layer2_delta)
    weight0 += layer0.T.dot(layer1_delta)

