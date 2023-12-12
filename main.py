import numpy as np
import matplotlib.pyplot as plt
from qlassifier import MyClass


def main():
    # ==================================
    # Inizializzazione
    # ==================================
    nome_file = "mnist_reuploading.txt"
    train_size = 5
    epochs = 2
    batch_size = 2
    resize = 10
    filt = "no"
    method = "sgd"

    my_class = MyClass(
        train_size=train_size,
        resize=resize,
        filt=filt,
        epochs=epochs,
        batch_size=batch_size,
        method=method,
    )
    my_class.initialize_data()

    best, params, extra = my_class.training_loop()

    print("Fine del training")
    with open(nome_file, "a") as file:
        print("/" * 60, file=file)
        print("/" * 60, file=file)
        print("Fine del training", file=file)
        print("/" * 60, file=file)
        print("/" * 60, file=file)
        print(f"Parametri finali circuito", file=file)
        print(f"{vparams}", file=file)
        print(f"Loss {best}", file=file)
        print("=" * 60, file=file)


if __name__ == "__main__":
    main()
