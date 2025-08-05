# nn_batch.py
# Python 3.x

import numpy as np
import random
import math
# import sys

# helper functions


def makeData(numFeatures, numHidden, numClasses, numRows, nn_seed):

    nn = NeuralNetwork(numFeatures, numHidden, numClasses, nn_seed)
    numWts = nn.totalWeights(numFeatures, numHidden, numClasses)
    wts = np.zeros(numWts, dtype=np.float32)
    w_lo = -9.0
    w_hi = +9.0
    for i in range(0, numWts):
        wts[i] = (w_hi - w_lo) * nn.rnd.random() + w_lo
    nn.setWeights(wts)

    numCols = numFeatures + numClasses
    result = np.zeros(shape=[numRows, numCols], dtype=np.float32)
    x_lo = -5.0
    x_hi = +5.0
    for i in range(0, numRows):  # each row
        x_vals = np.zeros(numFeatures, dtype=np.float32)
        y_vals = np.zeros(numClasses, dtype=np.float32)
        for j in range(0, numFeatures):
            x_vals[j] = (x_hi - x_lo) * nn.rnd.random() + x_lo
        y_vals = nn.computeOutputs(x_vals)
        # input("Press Enter to continue...")

        for j in range(0, numFeatures):
            result[i, j] = x_vals[j]  # insert x_values into result

        idx = np.argmax(y_vals)  # find the '1' cell
        t_vals = np.zeros(numClasses, dtype=np.float32)
        t_vals[idx] = 1.0
        for j in range(0, numClasses):
            result[i, j+numFeatures] = t_vals[j]

    return result


def splitData(data, trainPct):
    numTrainRows = int(len(data) * trainPct)
    numTestRows = len(data) - numTrainRows
    trainData = data[range(0, numTrainRows),]
    testData = data[range(numTrainRows, len(data)),]
    return (trainData, testData)


def showVector(v, dec):
    fmt = "%." + str(dec) + "f"  # like %.4f
    for i in range(len(v)):
        x = v[i]
        if x >= 0.0:
            print(' ', end='')
        print(fmt % x + '  ', end='')
    print('')

# def showMatrix(m, dec):
#   fmt = "%." + str(dec) + "f" # like %.4f
#   for i in range(len(m)):
#     for j in range(len(m[i])):
#       x = m[i,j]
#       if x >= 0.0: print(' ', end='')
#       print(fmt % x + '  ', end='')
#     print('')


def showMatrixPartial(m, numRows, dec, indices):
    fmt = "%." + str(dec) + "f"  # like %.4f
    lastRow = len(m) - 1
    width = len(str(lastRow))
    for i in range(numRows):
        if indices == True:
            print("[", end='')
            print(str(i).rjust(width), end='')
            print("] ", end='')

        for j in range(len(m[i])):
            x = m[i, j]
            if x >= 0.0:
                print(' ', end='')
            print(fmt % x + '  ', end='')
        print('')
    print(" . . . ")

    if indices == True:
        print("[", end='')
        print(str(lastRow).rjust(width), end='')
        print("] ", end='')
    for j in range(len(m[lastRow])):
        x = m[lastRow, j]
        if x >= 0.0:
            print(' ', end='')
        print(fmt % x + '  ', end='')
    print('')

# -----


class NeuralNetwork:

    def __init__(self, numInput, numHidden, numOutput, seed):
        self.ni = numInput
        self.nh = numHidden
        self.no = numOutput

        self.iNodes = np.zeros(shape=[self.ni], dtype=np.float32)
        self.hNodes = np.zeros(shape=[self.nh], dtype=np.float32)
        self.oNodes = np.zeros(shape=[self.no], dtype=np.float32)

        self.ihWeights = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)
        self.hoWeights = np.zeros(shape=[self.nh, self.no], dtype=np.float32)

        self.hBiases = np.zeros(shape=[self.nh], dtype=np.float32)
        self.oBiases = np.zeros(shape=[self.no], dtype=np.float32)

        self.rnd = random.Random(seed)  # allows multiple instances
        self.initializeWeights()

    def setWeights(self, weights):
        if len(weights) != self.totalWeights(self.ni, self.nh, self.no):
            print("Warning: len(weights) error in setWeights()")

        idx = 0
        for i in range(self.ni):
            for j in range(self.nh):
                self.ihWeights[i, j] = weights[idx]
                idx += 1

        for j in range(self.nh):
            self.hBiases[j] = weights[idx]
            idx += 1

        for j in range(self.nh):
            for k in range(self.no):
                self.hoWeights[j, k] = weights[idx]
                idx += 1

        for k in range(self.no):
            self.oBiases[k] = weights[idx]
            idx += 1

    def getWeights(self):
        tw = self.totalWeights(self.ni, self.nh, self.no)
        result = np.zeros(shape=[tw], dtype=np.float32)
        idx = 0  # points into result

        for i in range(self.ni):
            for j in range(self.nh):
                result[idx] = self.ihWeights[i, j]
                idx += 1

        for j in range(self.nh):
            result[idx] = self.hBiases[j]
            idx += 1

        for j in range(self.nh):
            for k in range(self.no):
                result[idx] = self.hoWeights[j, k]
                idx += 1

        for k in range(self.no):
            result[idx] = self.oBiases[k]
            idx += 1

        return result

    def initializeWeights(self):
        numWts = self.totalWeights(self.ni, self.nh, self.no)
        wts = np.zeros(shape=[numWts], dtype=np.float32)
        lo = -0.01
        hi = 0.01
        for idx in range(len(wts)):
            wts[idx] = (hi - lo) * self.rnd.random() + lo
        self.setWeights(wts)

    def computeOutputs(self, xValues):
        hSums = np.zeros(shape=[self.nh], dtype=np.float32)
        oSums = np.zeros(shape=[self.no], dtype=np.float32)

        for i in range(self.ni):
            self.iNodes[i] = xValues[i]

        for j in range(self.nh):
            for i in range(self.ni):
                hSums[j] += self.iNodes[i] * self.ihWeights[i, j]

        for j in range(self.nh):
            hSums[j] += self.hBiases[j]

        for j in range(self.nh):
            self.hNodes[j] = self.hypertan(hSums[j])

        for k in range(self.no):
            for j in range(self.nh):
                oSums[k] += self.hNodes[j] * self.hoWeights[j, k]

        for k in range(self.no):
            oSums[k] += self.oBiases[k]

        softOut = self.softmax(oSums)
        for k in range(self.no):
            self.oNodes[k] = softOut[k]

        result = np.zeros(shape=self.no, dtype=np.float32)
        for k in range(self.no):
            result[k] = self.oNodes[k]

        return result

    def trainOnline(self, trainData, maxEpochs, learnRate):
        # online with tanh + softmax & error
        # hidden-to-output weights gradients
        hoGrads = np.zeros(shape=[self.nh, self.no], dtype=np.float32)
        # output node biases gradients
        obGrads = np.zeros(shape=[self.no], dtype=np.float32)
        # input-to-hidden weights gradients
        ihGrads = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)
        # hidden biases gradients
        hbGrads = np.zeros(shape=[self.nh], dtype=np.float32)

        # output signals: gradients w/o assoc. input terms
        oSignals = np.zeros(shape=[self.no], dtype=np.float32)
        # hidden signals: gradients w/o assoc. input terms
        hSignals = np.zeros(shape=[self.nh], dtype=np.float32)

        epoch = 0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)
        numTrainItems = len(trainData)
        # [0, 1, 2, . . n-1]  # rnd.shuffle(v)
        indices = np.arange(numTrainItems)

        while epoch < maxEpochs:
            self.rnd.shuffle(indices)  # scramble order of training items
            for ii in range(numTrainItems):
                idx = indices[ii]

                for j in range(self.ni):
                    x_values[j] = trainData[idx, j]  # get the input values
                for j in range(self.no):
                    # get the target values
                    t_values[j] = trainData[idx, j+self.ni]
                self.computeOutputs(x_values)  # results stored internally

                # 1. compute output node signals
                for k in range(self.no):
                    derivative = (1 - self.oNodes[k]) * \
                        self.oNodes[k]  # softmax
                    # E=(t-o)^2 do E'=(o-t)
                    oSignals[k] = derivative * (self.oNodes[k] - t_values[k])

                # 2. compute hidden-to-output weight gradients using output signals
                for j in range(self.nh):
                    for k in range(self.no):
                        hoGrads[j, k] = oSignals[k] * self.hNodes[j]

                # 3. compute output node bias gradients using output signals
                for k in range(self.no):
                    # 1.0 dummy input can be dropped
                    obGrads[k] = oSignals[k] * 1.0

                # 4. compute hidden node signals
                for j in range(self.nh):
                    sum = 0.0
                    for k in range(self.no):
                        sum += oSignals[k] * self.hoWeights[j, k]
                    # tanh activation
                    derivative = (1 - self.hNodes[j]) * (1 + self.hNodes[j])
                    hSignals[j] = derivative * sum

                # 5 compute input-to-hidden weight gradients using hidden signals
                for i in range(self.ni):
                    for j in range(self.nh):
                        ihGrads[i, j] = hSignals[j] * self.iNodes[i]

                # 6. compute hidden node bias gradients using hidden signals
                for j in range(self.nh):
                    # 1.0 dummy input can be dropped
                    hbGrads[j] = hSignals[j] * 1.0

                # update weights and biases using the gradients

                # 1. update input-to-hidden weights
                for i in range(self.ni):
                    for j in range(self.nh):
                        delta = -1.0 * learnRate * ihGrads[i, j]
                        self.ihWeights[i, j] += delta

                # 2. update hidden node biases
                for j in range(self.nh):
                    delta = -1.0 * learnRate * hbGrads[j]
                    self.hBiases[j] += delta

                # 3. update hidden-to-output weights
                for j in range(self.nh):
                    for k in range(self.no):
                        delta = -1.0 * learnRate * hoGrads[j, k]
                        self.hoWeights[j, k] += delta

                # 4. update output node biases
                for k in range(self.no):
                    delta = -1.0 * learnRate * obGrads[k]
                    self.oBiases[k] += delta

            epoch += 1

            if epoch % 25 == 0:
                mse = self.meanSquaredError(trainData)
                print("epoch = " + str(epoch) + " ms error = %0.4f " % mse)

        # end while

        result = self.getWeights()
        return result
    # end trainOnline

    # ----------------

    def trainBatch(self, trainData, maxEpochs, learnRate):
        # full batch with tanh + softmax & ms error
        # this version accumulates gradients instead of deltas

        # hidden-to-output weights gradients
        hoGrads = np.zeros(shape=[self.nh, self.no], dtype=np.float32)
        # output node biases gradients
        obGrads = np.zeros(shape=[self.no], dtype=np.float32)
        # input-to-hidden weights gradients
        ihGrads = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)
        # hidden biases gradients
        hbGrads = np.zeros(shape=[self.nh], dtype=np.float32)

        # output signals: gradients w/o assoc. input terms
        oSignals = np.zeros(shape=[self.no], dtype=np.float32)
        # hidden signals: gradients w/o assoc. input terms
        hSignals = np.zeros(shape=[self.nh], dtype=np.float32)

        epoch = 0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)
        numTrainItems = len(trainData)
        # [0, 1, 2, . . n-1]  # rnd.shuffle(v)
        indices = np.arange(numTrainItems)

        while epoch < maxEpochs:
            # self.rnd.shuffle(indices)  # scramble order of training items -- not necessary for full-batch
            # zero-out batch training accumulated weight and bias gradients
            # input-to-hidden weights accumulated grads
            ihWtsAccGrads = np.zeros(
                shape=[self.ni, self.nh], dtype=np.float32)
            # hidden biases accumulated grads
            hBiasesAccGrads = np.zeros(shape=[self.nh], dtype=np.float32)
            hoWtsAccGrads = np.zeros(
                shape=[self.nh, self.no], dtype=np.float32)  # hidden-to-output
            oBiasesAccGrads = np.zeros(
                shape=[self.no], dtype=np.float32)  # output node biases

            # visit each item, accumulate weight grads but don't update
            for ii in range(numTrainItems):
                idx = indices[ii]  # note these are unscrambled

                for j in range(self.ni):
                    x_values[j] = trainData[idx, j]  # get the input values
                for j in range(self.no):
                    # get the target values
                    t_values[j] = trainData[idx, j+self.ni]
                self.computeOutputs(x_values)  # results stored internally

                # 1. compute output node signals
                for k in range(self.no):
                    derivative = (1 - self.oNodes[k]) * \
                        self.oNodes[k]  # softmax
                    # E=(t-o)^2 do E'=(o-t)
                    oSignals[k] = derivative * (self.oNodes[k] - t_values[k])

                # 2. compute hidden-to-output weight grads and accumulate
                for j in range(self.nh):
                    for k in range(self.no):
                        hoGrads[j, k] = oSignals[k] * self.hNodes[j]
                        hoWtsAccGrads[j, k] += hoGrads[j, k]

                # 3. compute and accumulate output node bias gradients
                for k in range(self.no):
                    # 1.0 dummy input can be dropped
                    obGrads[k] = oSignals[k] * 1.0
                    oBiasesAccGrads[k] += obGrads[k]

                # 4. compute hidden node signals
                for j in range(self.nh):
                    sum = 0.0
                    for k in range(self.no):
                        sum += oSignals[k] * self.hoWeights[j, k]
                    # tanh activation
                    derivative = (1 - self.hNodes[j]) * (1 + self.hNodes[j])
                    hSignals[j] = derivative * sum

                # 5 compute and accumulate input-to-hidden weight gradients using hidden signals
                for i in range(self.ni):
                    for j in range(self.nh):
                        ihGrads[i, j] = hSignals[j] * self.iNodes[i]
                        ihWtsAccGrads[i, j] += ihGrads[i, j]

                # 6. compute and accumulate hidden node bias gradients using hidden signals
                for j in range(self.nh):
                    # 1.0 dummy input can be dropped
                    hbGrads[j] = hSignals[j] * 1.0
                    hBiasesAccGrads[j] += hbGrads[j]

            # end-for (all training items)

            # now do the updates - calculate deltas using accumulated gradients
            # 1. update input-to-hidden weights
            for i in range(self.ni):
                for j in range(self.nh):
                    delta = -1.0 * learnRate * ihWtsAccGrads[i, j]
                    self.ihWeights[i, j] += delta

            # 2. update hidden node biases
            for j in range(self.nh):
                delta = -1.0 * learnRate * hBiasesAccGrads[j]
                self.hBiases[j] += delta

            # 3. update hidden-to-output weights
            for j in range(self.nh):
                for k in range(self.no):
                    delta = -1.0 * learnRate * hoWtsAccGrads[j, k]
                    self.hoWeights[j, k] += delta

            # 4. update output node biases
            for k in range(self.no):
                delta = -1.0 * learnRate * oBiasesAccGrads[k]
                self.oBiases[k] += delta

            epoch += 1

            if epoch % 25 == 0:
                mse = self.meanSquaredError(trainData)
                print("epoch = " + str(epoch) + " ms error = %0.4f " % mse)

        # end while

        result = self.getWeights()
        return result
    # end trainBatch

    def accuracy(self, tdata):  # train or test data matrix
        num_correct = 0
        num_wrong = 0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)

        for i in range(len(tdata)):  # walk thru each data item
            for j in range(self.ni):  # peel off input values from curr data row
                x_values[j] = tdata[i, j]
            for j in range(self.no):  # peel off tareget values from curr data row
                t_values[j] = tdata[i, j+self.ni]

            y_values = self.computeOutputs(x_values)  # computed output values)
            max_index = np.argmax(y_values)  # index of largest output value

            if abs(t_values[max_index] - 1.0) < 1.0e-5:
                num_correct += 1
            else:
                num_wrong += 1

        return (num_correct * 1.0) / (num_correct + num_wrong)

    def meanSquaredError(self, tdata):  # on train or test data matrix
        sumSquaredError = 0.0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)

        for ii in range(len(tdata)):  # walk thru each data item
            for jj in range(self.ni):  # peel off input values from curr data row
                x_values[jj] = tdata[ii, jj]
            for jj in range(self.no):  # peel off tareget values from curr data row
                t_values[jj] = tdata[ii, jj+self.ni]

            y_values = self.computeOutputs(x_values)  # computed output values

            for j in range(self.no):
                # target - output to be consistent
                err = t_values[j] - y_values[j]
                sumSquaredError += err * err  # (t-o)^2

        return sumSquaredError / len(tdata)

    @staticmethod
    def hypertan(x):
        if x < -20.0:
            return -1.0
        elif x > 20.0:
            return 1.0
        else:
            return math.tanh(x)

    @staticmethod
    def softmax(oSums):
        result = np.zeros(shape=[len(oSums)], dtype=np.float32)
        m = max(oSums)
        divisor = 0.0
        for k in range(len(oSums)):
            divisor += math.exp(oSums[k] - m)
        for k in range(len(result)):
            result[k] = math.exp(oSums[k] - m) / divisor
        return result

    @staticmethod
    def totalWeights(nInput, nHidden, nOutput):
        tw = (nInput * nHidden) + (nHidden * nOutput) + nHidden + nOutput
        return tw

# end class NeuralNetwork


def main():
    print("\nBegin NN batch train demo \n")

    numRows = 1000
    print("Generating " + str(numRows) + " rows of synthetic data \n")
    allData = makeData(4, 5, 3, numRows, nn_seed=1)
    print("Splitting data into 80%-20% train-test \n")
    (trainData, testData) = splitData(allData, trainPct=0.80)
    print("Training data: ")
    # x rows, 4 decimals, show indices
    showMatrixPartial(trainData, 3, 4, True)
    print("\nTest data: ")
    showMatrixPartial(testData, 3, 4, True)  # x rows, 4 decimals

    numInput = 4
    numHidden = 7
    numOutput = 3
    print("\nCreating a %d-%d-%d neural network " %
          (numInput, numHidden, numOutput))
    nn = NeuralNetwork(numInput, numHidden, numOutput, seed=13)

    maxEpochs = 100
    learnRate = 0.01
    print("\nSetting maxEpochs = " + str(maxEpochs))
    print("Setting learning rate = %0.3f " % learnRate)

    print("Starting training (batch - accumulated gradients)")
    nn.trainBatch(trainData, maxEpochs, learnRate)
    print("Training complete")

    accTrain = nn.accuracy(trainData)
    accTest = nn.accuracy(testData)

    print("Accuracy on train data = %0.4f " % accTrain)
    print("Accuracy on test data  = %0.4f " % accTest)

    print("-------------")

    print("\nRe-creating a %d-%d-%d neural network " %
          (numInput, numHidden, numOutput))
    nn = NeuralNetwork(numInput, numHidden, numOutput, seed=13)

    learnRate = 0.01  # not necessarily optimum

    print("Starting training (online)")
    nn.trainOnline(trainData, maxEpochs, learnRate)
    print("Training complete")

    accTrain = nn.accuracy(trainData)
    accTest = nn.accuracy(testData)

    print("Accuracy on train data = %0.4f " % accTrain)
    print("Accuracy on test data  = %0.4f " % accTest)

    print("\nEnd demo ")


if __name__ == "__main__":
    main()

# end script
