import pandas


def get_names(table):
    """
    Функция возращает заголовки CSV таблицы
    :param table: pandas.read_csv() object
    :return: list
    """
    return table.columns.values


def get_row(table, row):
    """
    Функция возвращает определенную строку
    :param table: pandas.read_csv() object
    :param row: int -> row id
    :return: list
    """
    return table.values[:][row]


def get_first(table, val):
    """
    Функция возвращает первые N строк
    :param table: pandas.readcsv() object
    :param val: int -> сколько строк возвращаем
    :return: list
    """
    return table.head(val)


def get_column(table, column):
    """
    Функция возвращает колонку
    :param table: pandas.readcsv() object
    :param column: int -> column id
    :return: list
    """
    rows = []
    for row in table.values:

        rows.append(row[column])

    return rows


if __name__ == "__main__":
    csv_table = pandas.read_csv("multidimensional_data.csv", sep=";")  # Загружаем CSV таблицу, параметр1 = путь к таблице, sep=";" разделитель используемый в таблице

    print("Выводим полную таблицу\n", csv_table)
    print("Выводим заголовки\n", get_names(csv_table))
    print("Получаем первые N строк\n", get_first(csv_table, 5))
    print("Получаем строку\n", get_row(csv_table, 7))
    print("Получаем колонку\n", get_column(csv_table, 3))
