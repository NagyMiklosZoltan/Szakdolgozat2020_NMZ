import glob
import numpy as np
import matplotlib.pyplot as plt


def plot_history(history):
    # Get training and test loss histories
    # training_loss = history.history['loss'][2:]
    # test_loss = history.history['val_loss'][2:]
    #
    # # Create count of the number of epochs
    #  epoch_count = range(1, len(training_loss) + 1)
    #
    # # Visualize loss history
    # plt.plot(epoch_count, training_loss, 'r--')
    # plt.plot(epoch_count, test_loss, 'b-')
    # plt.legend(['Training loss', 'Test loss'])
    # plt.xlabel('Epoch')
    # plt.ylabel('loss')
    # plt.yscale('log')
    # plt.show()

    training_accuracy = history.history['acc']
    test_accuracy = history.history['val_acc']

    epoch_count = range(1, len(training_accuracy) + 1)

    # Visualize accurary history
    plt.plot(epoch_count, training_accuracy, 'r--')
    plt.plot(epoch_count, test_accuracy, 'b-')
    plt.legend(['Training accuracy', 'Test acc'])
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.ylim(0.0, 1.0)
    plt.show()
