# coding=utf-8
__author__ = 'Gyo'

import theano
import theano.tensor as T

class LogisticRegressionLayer(object):
    def __init__(self, layer_name, input_image,initial_filter_values, initial_bias_values):

        self.Filter = theano.shared(
            value= initial_filter_values,
            name='Filter_'+layer_name
        )

        self.Bias = theano.shared(
            value=initial_bias_values,
            name='Bias_' + layer_name
        )

        self.ProductoCruz = T.dot(input_image, self.Filter) + self.Bias

        """
        ### T.nnet.softmax(self.ProductoCruz) ###
        para cada ejemplo(row) de self.ProductoCruz, por cada uno de sus clases(columnas),
        produce una probabilidad con respecto de sus otras clases calculadas
        result: NoRows X NoTotalClasses
        """
        self.p_y_given_x = T.nnet.softmax(self.ProductoCruz)

        """
        ### T.argmax(self.p_y_given_x, axis=1) ###
        A partir de las probabilidades calculadas de cada imagen(self.p_y_given_x),
        obtenemos un arreglo de tamaño NoEjemplos X 1,
        donde cada valor contiene el indice(Clase) de la probabilidad mas alta,
        es decir la clase calculada o predecida
        """
        self.y_predictions = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, correct_classes_for_y):
        """
        :param correct_y: es una lista de enteros donde cada indice corresponde a un ejemplo y su valor corresponde a la clase correct
        :return:

        Esta funcion es nuestro criterio de medida de que tan bien ha realizado el calculo la CNN,
        Checamos la probabilidad predecida para Y, despues sumamos todas las probabilidades y les sacamos el promedio de que tanto se equivoca
        """
        examples_ids = T.arange(correct_classes_for_y.shape[0]) #creamos una secuencia de 0 hasta el numero de de elementos en y

        #result_: Size: noRows X NoTotalClasses, cada valor es la probabilidad (en log) calculada del NRow para la NClase
        prbabilitiesLog_of_predicted_classes = T.log(self.p_y_given_x)  #aplicamos la funcion log a las probabilidades predecidas por cada una de las posibles clases, Siempre resultara un numero negativo, si la probabilidad es muy baja dara un resultado mas negativo

        calculatedProbs_for_correct_class = prbabilitiesLog_of_predicted_classes[examples_ids, correct_classes_for_y]  #por cada row(ejemplo predecido) obtenemos la probabilidad de la clase correcta(y), si es correcta debe ser muy alta y si es incorrecta debe ser muy baja

        return T.mean(calculatedProbs_for_correct_class) #Regresamos el promedio de las respuestas calculadas, si es muy alto significa que va bien por lo que queremos Maximizar el resultado

    def errors(self, correct_classes_for_y):
        """
        Regresa el promedio de errores, el resultado esta en el rango DE 0 a 1 donde 0 significa que no hubo error y 1 significa que en todos hubo error
        :param y:
        :return:
        """
        result = T.neq(self.y_predictions, correct_classes_for_y)  # the T.neq operator returns a vector of 0s and 1s, where 1 represents a mistake in prediction

        return T.mean(result)
