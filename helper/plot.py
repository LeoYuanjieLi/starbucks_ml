import matplotlib.pyplot as plt


def plot_hist(dataframe):
    for i in range(0, len(dataframe.columns), 2):
        n_bins = 20
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(20, 5))

        x = dataframe[dataframe.columns[i]]
        x = x.dropna()
        axs[0].hist(x, bins=n_bins)
        axs[0].set_title(dataframe.columns[i])

        if i + 1 < len(dataframe.columns):
            _y = dataframe[dataframe.columns[i + 1]]
            axs[1].hist(_y, bins=n_bins)
            axs[1].set_title(dataframe.columns[i + 1])

    plt.show()
