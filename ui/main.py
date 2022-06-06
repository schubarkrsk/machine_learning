# Основные библиотеки
import pandas
from matplotlib import pyplot
import numpy as np

# Библиотеки датасетов
from sklearn.datasets import load_boston


def draw_boxplot():
    """
    Функция отрисовывает столбчатую диаграмму с усами
    на основе заданного URL адреса
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    table = pandas.read_csv(url, names=names)

    table.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    pyplot.show()


def example_sin():
    """
    Функция взята их документации PyPlot
    в связке с NumPy она отрисовывает синусоиду (график по точкам)
    """
    x = np.arange(0, 5, 0.1)
    y = np.sin(x)
    pyplot.plot(x, y)
    pyplot.show()


def example_pie():
    """
    Функция отрисовывает круговую диаграмму на основе
    CSV таблицы, беря из нее первые 10 строк и значений столбца № 3 (row[2])
    подписи беруться из столбца № 0
    """
    table = pandas.read_csv("multidimensional_data.csv", sep=";")
    x = []
    labels = []
    for row in table.head(10).values:
        labels.append(row[0])
        x.append(row[2])

    pyplot.pie(x, labels=labels)
    pyplot.show()


def draw():
    """
    Функция в процессе разработки
    """
    boston = load_boston()


if __name__ == "__main__":
    draw_boxplot()
    example_sin()
    example_pie()

    # Примеры можно найти в документации на оф. сайте
    # https//matplotlib.org/stable/api/_as_gen/ matplotlib.pyplot.html
