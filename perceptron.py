import numpy as np


class Perceptron:
    """A simple linear classification model y=sign(w·x+b).

    Properties:
        x: list
            Feature vector used to classify instances.
        y: list
            Label vector used to mark categories. The value in list must be **-1** or **+1**.
        weight: float or list, default=0
            Weight vector presented as w in the formula.
        bias: float, default=0
            Bias scalar presented as b in the formula.
    """
    def __init__(self, x: list, y: list, *, weight=0, bias=0):
        self.x = np.array(x)
        self.y = np.array(y)
        self.weight = np.array([weight for _ in range(len(x[0]))])
        self.bias = bias

    def calculate_sgd(self, step=1):
        """Calculate parameters of perceptron through stochastic gradient
        descent algorithm.

        Parameters:
            step: float, default=1
                Step is learning rate in every iteration cycle. The value of
                step must in  (0, 1].
        Return:
            Perceptron
        """
        point_index = 0
        w = self.weight
        b = self.bias

        def exist_mistaken_point(w, b):
            """Judge if exist a mistaken point using this perceptron model.

            The judge condition is y_i(w·x+b)<=0. This condition is used in
            traditional stochastic gradient descent algorithm.
            """
            if self.y[point_index] * (np.dot(w, self.x[point_index]) + b) <= 0:
                return True
            else:
                return False

        def update_parameter(w, b):
            w = w + step * self.x[point_index] * self.y[point_index]
            b = b + step * self.y[point_index]
            return w, b

        while point_index < len(self.x):
            if exist_mistaken_point(w, b):
                w, b = update_parameter(w, b)
                point_index = 0
            else:
                point_index += 1
        self.weight = w
        self.bias = b
        return self

    def calculate_dual_sgd(self, step=1, a=0):
        """Calculate parameters of perceptron through dual algorithm of stochastic gradient descent.

        Parameters:
            step: float, default=1
                Step is learning rate in every iteration cycle. The value of step must in  (0, 1].
            a: float or list, default=0
                Parameter a is one of perceptron parameters. It has the same meaning with parameter w.
        Return:
            Perceptron

        """
        a = np.array([a for _ in range(len(self.x))])
        b = self.bias
        point_index = 0

        def get_gram_matrix():
            """Gram matrix is a square matrix. It saves all the values of x_i·x_j."""
            gram = np.zeros((len(self.x), len(self.x)))
            for i in range(len(self.x)):
                for j in range(len(self.x)):
                    gram[i][j] = np.dot(self.x[i], self.x[j])

            return gram

        gram = get_gram_matrix()

        def exist_mistaken_point(point_index, a, b, gram):
            """Judge if exist a mistaken point using this perceptron model.

            The judge condition is y_i((sum(a_j*y_j*gram[j][i])+b)<=0. This
            condition is used in dual algorithm of stochastic gradient descent.
            """
            sum = 0
            for j in range(len(self.x)):
                sum += a[j] * self.y[j] * gram[j][point_index]

            sum = self.y[point_index] * (sum + b)

            if sum <= 0:
                return True
            else:
                return False

        def update_parameter(a, b):
            a[point_index] = a[point_index] + step
            b = b + step * self.y[point_index]

            return a, b

        def transform_a_to_w(a):
            w = np.zeros(self.x[0].shape)
            for i in range(len(self.x)):
                w += a[i] * self.y[i] * self.x[i]

            return w

        while point_index < len(self.x):
            if exist_mistaken_point(point_index, a, b, gram):
                a, b = update_parameter(a, b)
                point_index = 0
            else:
                point_index += 1

        self.weight = transform_a_to_w(a)
        self.bias = b
        return self

    def plot_line(self, x_list: list):
        """Draw classification line of perceptron.

        Parameters:
            x_list: list
                Points position in x0(x)-axis direction.
        Return:
            y_list: list
                Points position in x1(y)-axis direction.
        """
        y_list=[]
        for x in x_list:
            y = -(self.weight[0] * x + self.bias) / self.weight[1]
            y_list.append(y)

        return y_list

