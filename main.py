import numpy as np
import pickle
import matplotlib.pyplot as plt
from qlassifier import MyClass
from help_functions import plot_metrics


def main():
    # ==================================
    # Inizializzazione
    # ==================================
    nome_file = "mnist_reuploading.txt"
    resize = 10
    filt = "no"
    train_size = 9
    epochs = 2
    learning_rate = 0.1
    method = "Adadelta"
    batch_size = 3

    my_class = MyClass(
        train_size=train_size,
        resize=resize,
        filt=filt,
        epochs=epochs,
        batch_size=batch_size,
        method=method,
        learning_rate=learning_rate,
    )
    my_class.initialize_data()

    epoch_loss, params, extra = my_class.training_loop()
    plot_metrics(epochs, epoch_loss)

    # Save final parameters
    with open("saved_parameters.pkl", "wb") as f:
        pickle.dump(self.vparams, f, pickle.HIGHEST_PROTOCOL)

    print("Fine del training")
    with open(nome_file, "a") as file:
        print("/" * 60, file=file)
        print("/" * 60, file=file)
        print("Fine del training", file=file)
        print("/" * 60, file=file)
        print("/" * 60, file=file)
        print(f"Parametri finali circuito", file=file)
        print(f"{params}", file=file)
        print(f"Loss {best}", file=file)
        print("=" * 60, file=file)


if __name__ == "__main__":
    main()
