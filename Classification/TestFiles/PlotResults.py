import glob
import numpy as np
import matplotlib.pyplot as plt


def plot_history(history):
    # Get training and test loss histories
    training_loss = history.history['loss'][2:]
    test_loss = history.history['val_loss'][2:]

    # training_accuracy = history.history['accuracy']
    # test_accuracy = history.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training loss', 'Test loss'])
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.show()

    # # Visualize loss history
    # plt.plot(epoch_count, training_accuracy, 'r--')
    # plt.plot(epoch_count, test_accuracy, 'b-')
    # plt.legend(['Training accuracy', 'Test acc'])
    # plt.xlabel('Epoch')
    # plt.ylabel('accuracy')
    # plt.show()


    #
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # t = f.suptitle('Basic CNN Performance', fontsize=12)
    # f.subplots_adjust(top=0.85, wspace=0.3)
    #
    # epoch_list = list(range(1, 31))
    # ax1.plot(epoch_list, history['accuracy'], label='Train Accuracy')
    # ax1.plot(epoch_list, history['val_accuracy'], label='Validation Accuracy')
    # ax1.set_xticks(np.arange(0, 31, 5))
    # ax1.set_ylabel('Accuracy Value')
    # ax1.set_xlabel('Epoch')
    # ax1.set_title('Accuracy')
    # l1 = ax1.legend(loc="best")
    #
    # ax2.plot(epoch_list, history['loss'], label='Train Loss')
    # ax2.plot(epoch_list, history['val_loss'], label='Validation Loss')
    # ax2.set_xticks(np.arange(0, 31, 5))
    # ax2.set_ylabel('Loss Value')
    # ax2.set_xlabel('Epoch')
    # ax2.set_title('Loss')
    # l2 = ax2.legend(loc="best")