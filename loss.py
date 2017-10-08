from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''Your codes here'''
        batch_size = input.shape[0]
        size = input.size
        diff = (input - target)
        loss_values = np.zeros((1, batch_size))[0]
        loss_value = 0
        for i in xrange(batch_size):
            loss_values[i] = diff[i].dot(diff[i])
            loss_value += loss_values[i];
        loss_value /= (2.0 * size)
        # print loss_values
        # print loss_value
        # exit(0)
        return loss_value

    def backward(self, input, target):
        '''Your codes here'''
        return (input - target)

# class SoftmaxCrossEntropyLoss(object):
#     def __init__(self, name):
#         self.name = name

#     def forward(self, input, target):
#         i = 0;
#         length = len(input)
#         loss_value = 0
#         for i in xrange(length):
#             loss_value -= target[i]*log(input)
#         return loss_value
#     def backward(self, input, target):
        