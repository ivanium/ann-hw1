import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        '''Your codes here'''
        self._saved_for_backward(input)
        return input * (input > 0)

    def backward(self, grad_output):
        '''Your codes here'''
        return (grad_output > 0)

class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        '''Your codes here'''
        output = 1 / (1 + np.exp(-input))
        self._saved_for_backward(output)
        return output

    def backward(self, grad_output):
        '''Your codes here'''
        prev_output = self._saved_tensor
        return prev_output * (1 - prev_output) * grad_output


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        '''Your codes here'''
        self._saved_for_backward(input)
        return input.dot(self.W) + self.b

    def backward(self, grad_output):
        '''Your codes here'''
        # compute grad and change self.diff_w
        input = self._saved_tensor
        batch_size, in_num = input.shape
        out_num = grad_output.shape[1]

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)
        
        for i in xrange(batch_size):
            self.grad_W += input[i].reshape(in_num, 1).dot(grad_output[i].reshape(1,out_num))
            self.grad_b += grad_output[i]
        return grad_output.dot(self.W.T)

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
