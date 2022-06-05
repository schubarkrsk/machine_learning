# Основные библиотеки
import pandas
from matplotlib import pyplot

# Библиотеки датасетов
from sklearn.datasets import load_boston


def draw_boxplot():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    table = pandas.read_csv(url, names=names)

    table.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    pyplot.show()


def draw():
    boston = load_boston()


if __name__ == "__main__":
    draw_boxplot()
