import numpy as np
import pickle
import matplotlib.pyplot as plt
from qlassifier import MyClass
from help_functions import plot_metrics


def main():
    # ==================================
    # Inizializzazione
    # ==================================
    resize = 10
    filt = "no"
    train_size = 9
    epochs = 2
    learning_rate = 0.1
    method = "Adadelta"
    batch_size = 3

    # Create the class
    my_class = MyClass(
        train_size=train_size,
        resize=resize,
        filt=filt,
        epochs=epochs,
        batch_size=batch_size,
        method=method,
        learning_rate=learning_rate,
    )

    # Initialize data
    my_class.initialize_data()

    # Training
    epoch_train_loss, epoch_validation_loss, params, extra = my_class.training_loop()

    # Plot training and validation loss
    plot_metrics(epochs, epoch_loss)

    # Testing
    accuracy = my_class.test_loop()

    # Save final parameters
    with open("saved_parameters.pkl", "wb") as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

    # Print Accuracy
    with open("epochs.txt", "a") as file:
        print("/" * 60, file=file)
        print("/" * 60, file=file)
        print(f"Accuracy {accuracy.result().numpy()}", file=file)
        print("/" * 60, file=file)
        print("/" * 60, file=file)


if __name__ == "__main__":
    main()
