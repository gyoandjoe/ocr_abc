__author__ = 'Gyo'

import theano
import theano.tensor as T
import theano.sandbox.cuda.basic_ops as sbcuda
from theano.sandbox.cuda import dnn

class SoftMaxLayer(object):
    def __init__(self, layer_name, input_image, initial_filter_values, initial_bias_values):
        """
        :param layer_name:
        :param input_image: Image to be treated
        :param initial_filter_values: Initial values of filter
        :param initial_bias_values: Initial values of bias
        :param activation_function: function that computes the ProductoCruz
        :return:
        """

        self.LayerName = 'SoftMaxLayer_' + layer_name



        self.Bias = theano.shared(
            value=initial_bias_values,
            name='SoftMaxBias_' + str(self.LayerName),
        )

        self.Filter = theano.shared(
            value=initial_filter_values,
            name='SoftMaxFilter_'+str(self.LayerName),
            borrow = True
        )

        self.ProductoCruz = T.dot(input_image, self.Filter) + self.Bias

        self.p_y_given_x = T.nnet.softmax(self.ProductoCruz)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


    def negative_log_likelihood(self, y):
        """
        Esta funcion es nuestro criterio de medida de que tan bien ha realizado el calculo la CNN,
        Checamos la probabilidad predecida para Y, despues sumamos todas las probabilidades y les sacamos el promedio de que tanto se equivoca
        Cost Function with mean and not sum
        :param y: es una coleccion de indices donde cada row representa un ejemplo y su valor el indice correcto
        :return:
        """
        listaindices = T.arange(y.shape[0])  # creamos una secuencia de 0 hasta el numero de de elementos en y

        resultLog = T.log(
            self.p_y_given_x)  # aplicamos la funcion log a las probabilidades predecidas por cada una de las posibles clases, Siempre resultara un numero negativo, si la probabilidad es muy baja dara un resultado mas negativo
        result = resultLog[
            listaindices, y]  # por cada row(ejemplo predecido) obtenemos la probabilidad de la clase correcta(y), si es correcta debe ser muy alta y si es incorrecta debe ser muy baja

        return T.mean(
            result)  # Regresamos el promedio de las respuestas calculadas, si es muy alto significa que va bien por lo que queremos Maximizar el resultado


    def cost_function(self, true_y):
        return T.mean((T.nnet.categorical_crossentropy(self.p_y_given_x, true_y)))