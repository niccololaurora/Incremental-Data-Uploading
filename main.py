import numpy as np
import pickle
import matplotlib.pyplot as plt
from qclass import MyClass
from help_functions import plot_metrics


def main():
    # ==================================
    # Inizializzazione
    # ==================================
    nome_file = "epochs.txt"
    training_sample = 10
    epochs = 2
    learning_rate = 0.1
    method = "Adam"
    batch_size = 2
    layers = 2

    # Create the class
    my_class = MyClass(
        train_size=training_sample,
        epochs=epochs,
        batch_size=batch_size,
        method=method,
        learning_rate=learning_rate,
        nome_file=nome_file,
        layers=layers,
    )

    # Initialize data
    my_class.initialize_data()

    # Training
    epoch_train_loss, epoch_validation_loss, params, epochs = my_class.training_loop()

    # Plot training and validation loss
    plot_metrics(epochs, epoch_train_loss, method, epoch_validation_loss)

    # Testing
    accuracy = my_class.test_loop()

    # Save final parameters
    with open("saved_parameters.pkl", "wb") as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

    # Print Accuracy
    with open(nome_file, "a") as file:
        print("/" * 60, file=file)
        print("/" * 60, file=file)
        print(f"Accuracy {accuracy.result().numpy()}", file=file)
        print("/" * 60, file=file)
        print("/" * 60, file=file)


if __name__ == "__main__":
    main()
