from matplotlib import pyplot as plt


def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()


def plot_sample_length_distribution_longer_than(sample_texts, sample_lower_lim, sample_upper_lim):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    texts_longer_than_sample_lim = [s for s in sample_texts if sample_lower_lim <= len(s) <= sample_upper_lim]
    plt.hist([len(text) for text in texts_longer_than_sample_lim], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()


def plot_history(history_loss_train, history_loss_val):
    plt.plot(range(len(history_loss_train)), history_loss_train, label='train')
    plt.plot(range(len(history_loss_val)), history_loss_val, label='test')
    plt.legend()
    plt.show()
