import matplotlib.pyplot as plt

def plot_predictions(true_values, predicted_values):
    plt.figure(figsize=(10, 5))
    plt.plot(true_values, label="True Values")
    plt.plot(predicted_values, label="Predicted Values", linestyle="--")
    plt.legend()
    plt.show()
