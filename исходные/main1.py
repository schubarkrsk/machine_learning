# # Машинное обучение
# """
# список библиотек, которые необходимо установить:
#     scipy
#     numpy
#     matlibplot
#     pandas
#     sklearn
#     statsmodels
#     seaborn
#
# """
# # Проерка версий библиотек
import sys
# print('Python: {}'.format(sys.version))
#
# # Загрузка scipy
import scipy
# print('scipy: {}'.format(scipy.__version__))
#
# # Загрузка numpy
import numpy
# print('numpy: {}'.format(numpy.__version__))
#
# # Загрузка matplotlib
import matplotlib
# print('matplotlib: {}'.format(matplotlib.__version__))
#
# # Загрузка pandas
import pandas
# print('pandas: {}'.format(pandas.__version__))
#
# # Загрукзка scikit-learn
import sklearn
#
# # print('sklearn: {}'.format(sklearn.__version__))

# Загрузка библиотек
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import statsmodels.api as sm
import numpy as np
import seaborn
#
# # # Загрузка датасета цветов ирисов Фишера
# # url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# # names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# # dataset = read_csv(url, names=names)
# # print(dataset)
# # print(dataset.shape)
# # print(dataset.head(20))
# # print(dataset.describe())
# # print(dataset.groupby('class').size())
# #
# # #  Визуализация данных
# # dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# # pyplot.show()
# #
# # dataset.hist()
# # pyplot.show()
# #
# # scatter_matrix(dataset)
# # pyplot.show()
# #
# # # Разделение датасета на обучающую и контрольную выборки
# # array = dataset.values
# #
# # # Выбор первых 4-х столбцов
# # X = array[:,0:4]
# #
# # # Выбор 5-го столбца
# # y = array[:,4]
# #
# # # Разделение X и y на обучающую и контрольную выборки
# # X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# #
# # # Загружаем алгоритмы модели
# # models = []
# # models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# # models.append(('LDA', LinearDiscriminantAnalysis()))
# # models.append(('KNN', KNeighborsClassifier()))
# # models.append(('CART', DecisionTreeClassifier()))
# # models.append(('NB', GaussianNB()))
# # models.append(('SVM', SVC(gamma='auto')))
# #
# #
# # # оцениваем модель на каждой итерации
# # results = []
# # names = []
# #
# # for name, model in models:
# #     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
# #     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
# #     results.append(cv_results)
# #     names.append(name)
# #     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# #
# # pyplot.boxplot(results, labels=names)
# # pyplot.title('Сравнение алгоритмов')
# # pyplot.show()
# #
# # # Создаем прогноз на контрольной выборке
# # model = SVC(gamma='auto')
# # model.fit(X_train, Y_train)
# # predictions = model.predict(X_validation)
# #
# # # Оцениваем прогноз
# # print(accuracy_score(Y_validation, predictions))
# # print(confusion_matrix(Y_validation, predictions))
# # print(classification_report(Y_validation, predictions))
#
# # # =======================================================
# # #  Линейная регрессия
# # import numpy as np
# # from sklearn.linear_model import LinearRegression
# #
# # x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
# # y = np.array([5, 20, 14, 32, 22, 38])
# # print(x)
# # print(y)
# # model = LinearRegression()
# # model.fit(x, y)
# # model = LinearRegression().fit(x, y)
# # r_sq = model.score(x, y)
# # print('коэффициент детерминации:', round(r_sq,3))
# # b0=round(model.intercept_,3)
# # print("коэффициент b0:", b0)
# # b1=model.coef_
# # print("коэффициент b1:", b1)
# # #  Прогнозирование
# # y_pred = model.predict(x)
# # print('прогнозируемые значения:', y_pred, sep='\n')
# #
# # # Прогнозирование на дополнительных данных
# # x_new = np.arange(5).reshape((-1, 1))
# # print("дополнительные данные для прогноза",x_new, sep='\n')
# # y_new = model.predict(x_new)
# # print("значение прогноза", y_new, sep='\n')
# #
# # # ======================================================
#
# # Множественная регрессия
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# from sklearn.datasets import load_boston
# boston = load_boston()
# print(boston.DESCR)
# boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
# boston_df['MEDV'] = boston.target
# boston_df.head()
#
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]
# """
# Характеристики набора данных:
# :Количество экземпляров: 506
# :Количество атрибутов: 13 числовых/категориальных прогностических.
#  Медианное значение (атрибут 14) обычно является целевым значением.
# :Информация об атрибутах (по порядку):
# - CRIM Уровень преступности на душу населения в разбивке по городам
# - ZN доля жилой земли, зонированной на участки площадью более 25 000 кв.футов.
# - INDUS Доля промышленных площадей, не связанных с розничной торговлей, в расчете на город
# - CHAS Фиктивная переменная реки ЧАС Чарльз (= 1, если тракт граничит с рекой; 0 в противном случае)
# - NOX Концентрация оксидов азота NOX (частей на 10 миллионов)
# - RM Среднее количество комнат в среднем на одно жилое помещение
# - AGE ВОЗРАСТНАЯ доля квартир, занятых владельцами, построенных до 1940 года
# - DIS Взвешенные расстояния до пяти бостонских центров занятости
# - RAD Радный индекс доступности к радиальным магистралям
# - TAX НАЛОГ на недвижимость с полной стоимостью -ставка налога за 10 000 долларов США
# - PTRATIO Соотношение учащихся и учителей в разбивке по городам
# - B 1000(Bk - 0,63)^2, где Bk - доля чернокожих по городам
# - LSTAT % более низкий статус населения
# - MEDV Медианная стоимость домов, занятых владельцами, в 1000 долларов США
# :Отсутствующие Значения атрибутов: Отсутствуют
# : Создатель: Харрисон Д. и Рубинфельд Д.Л.
# Это копия набора данных UCI ML housing dataset.
# https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
# """
#
# #  Разведывательный анализ
#
# cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
# #  Визуализируем матрицу корреляций и ввиде тепловой карты:
# hm = sns.heatmap(boston_df[cols].corr(), cbar=True, annot=True)
# pyplot.show()
#
# # Определим зависимые и независимые переменные:
# X = boston_df[['LSTAT']].values
# y = boston_df['MEDV'].values
#
# from sklearn.linear_model import LinearRegression
# slr = LinearRegression()
# slr.fit(X, y)
# y_pred = slr.predict(X)
# print('Slope: {:.2f}'.format(slr.coef_[0]))
# print('Intercept: {:.2f}'.format(slr.intercept_))
#
# plt.scatter(X, y)
# plt.plot(X, slr.predict(X), color='red', linewidth=2)
# pyplot.show()
#
# # Для быстрой визуализации линейной зависимости можно также использовать функцию regplot из seaborn.
# sns.regplot(x="LSTAT", y="MEDV", data=boston_df)
# pyplot.show()
#
# #  линейная регрессионная модель
# print("линейная регрессионная модель")
# model = LinearRegression()
# model.fit(X, y)
# model = LinearRegression().fit(X, y)
# r_sq = model.score(X, y)
# print('коэффициент детерминации:', round(r_sq,3))
# b0=round(model.intercept_,3)
# print("коэффициент b0:", b0)
# b1=model.coef_
# print("коэффициент b1:", b1)
#
#
# # Проверка качества модели
# #  документация https://scikit-learn.org/stable/modules/classes.html#regression-metrics
# print('Проверка качества модели')
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=0)
# slr = LinearRegression()
#
# slr.fit(X_train, y_train)
# y_train_pred = slr.predict(X_train)
# y_test_pred = slr.predict(X_test)
#
# """
# Поскольку в модели несколько независимых переменных, мы не можем отобразить их зависимость на
# двумерном пространстве, но можем нанести на график связь между остатками модели и предсказанными
# значениями, что также поможет диагностировать качество модели.
# Это называется Residuals plot. C его помощью мы можем увидет нелинейность и выбросы, проверить
# случайность распределения ошибки.
# """
# from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
#
# print('ошибка СКО: {:.3f}, test: {:.3f}'.format(
#         mean_squared_error(y_train, y_train_pred),
#         mean_squared_error(y_test, y_test_pred)))
# print('коэффициент детерминации: {:.3f}, test: {:.3f}'.format(
#         r2_score(y_train, y_train_pred),
#         r2_score(y_test, y_test_pred)))
#
# #  График зависимости прогнозируемых значений от остатков
# plt.scatter(y_train_pred,  y_train_pred - y_train,
#             c='blue', marker='o', label='Обучающие данные')
# plt.scatter(y_test_pred,  y_test_pred - y_test,
#             c='lightgreen', marker='s', label='Тестовые данные')
# plt.xlabel('Прогнозируемые значения')
# plt.ylabel('Остатки')
# plt.legend(loc='upper left')
# plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
# plt.xlim([-10, 50])
# plt.tight_layout()
# pyplot.show()
#
# #  Statsmodels — ещё одна библиотека для построения статистических данных в Python,
# #  сравним реализацию линейных моделей в sklearn и statsmodels.
# print("линейная модель в statsmodels")
#
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
#
# results = smf.ols('MEDV ~ LSTAT', data=boston_df).fit()
# print(results.summary())
# print(results.params)
#
# # Регуляризация в sklearn
# """
# Регуляризация — это метод для уменьшения степени переобучения модели,
# разберемся в сути переобучения (overfitting).
# Переобучение дает неплавные кривые прогнозирования, т. е. «нерегулярные».
# Такие плохие сложные кривые прогнозирования обычно характеризуются весовыми значениями,
# которые имеют очень большие или очень малые величины.
# Поэтому один из способов уменьшить степень переобучения состоит в том, чтобы не допускать очень
# малых или больших весовых значений для модели.
#
# Линейная регрессия с большим числом предикторов – комплексная модель и характеризуется:
# Достаточно высоким смещением
# Высокой дисперсией
#
# Чем больше предикторов, тем больше риск переобучения модели.
# Переобучение также связано с размером коэфициентов.
#
# Переобучение – ситуация, в которой обучающая ошибка продолжает снижаться с повышением сложности
# модели, а тестовая ошибка растет.
#
# Как с этим бороться?
# Отбор наилучших предикторов
# Снижение размерности предикторов
# Регуляризация
#
# Регуляризация — это способ уменьшить сложность модели чтобы предотвратить переобучение или исправить
# некорректно поставленную задачу. Обычно это достигается добавлением некоторой априорной информации
# к условию задачи.
#
# В данном случае суть регуляризации состит в том, что мы создаём модель со всеми предикторами,
# а потом искуственно уменьшаем размер коэффициентов, прибавляя некоторую величину к ошибке.
#
# Ошибка — это то, что минимизируется обучением с помощью одного из примерно десятка численных
# методов вроде градиентного спуска (gradient descent), итерационного алгоритма Ньютона-Рафсона
# (iterative Newton-Raphson), L-BFGS, обратного распространения ошибок (back-propagation) и
# оптимизации роя (swarm optimization).
#
# Чтобы величины весовых значений модели не становились большими, процесс регуляризации штрафует
# весовые значения добавляя их в вычисление ошибки. Если весовые значения включаются в общую ошибку,
# которая минимизируется, тогда меньшие весовые значения будут давать меньшие значения ошибки.
#
# L1-регуляризация штрафует весовые значения добавлением суммы их абсолютных значений к ошибке.
# L2-регуляризация выполняет аналогичную операцию добавлением суммы их квадратов к ошибке.
# """
#
# """
# Ридж-регрессия или гребневая регрессия (ridge regression) - это один из методов понижения размерности.
# Часто его применяют для борьбы с переизбыточностью данных, когда независимые переменные коррелируют
# друг с другом (т.е. имеет место мультиколлинеарность). Следствием этого является плохая обусловленность
# матрицы X^T X и неустойчивость оценок коэффициентов регрессии.
# Оценки, например, могут иметь неправильный знак или значения, которые намного превосходят те,
# которые приемлемы из физических или практических соображений.
#
# Лассо регрессия (Least absolute shrinkage and selection operator) похожа на ридж регрессиюЮ но в ней
# штраф — это сумма модулей значений коэффициентов.
#
# В чем сила ридж и лассо?
# • Ридж регрессия снижает размер коэффициентов, а лассо сокращает многие до 0
# • Это позволяет снизить размерность (ридж) и выбрать важные предикторы (лассо)
# • Работает, когда p > n, где p — число предикторов
# • Работает, когда много коллинеарных предикторов
# • Обязательно надо делать шкалирование и центрирование, иначе предикторы с высоким стандартным
#   отклонением будут сильно штравоваться.
#
# ElasticNet — комбинация L1 и L2 регуляризации в разных пропорциях.
# """
#
# print('Регуляризация в sklearn')
# from sklearn.preprocessing import StandardScaler
#
# sc_x = StandardScaler()
# sc_y = StandardScaler()
# X_std = sc_x.fit_transform(X)
# y_std = sc_y.fit_transform(y.reshape(-1, 1)).flatten()
# # newaxis увеличивает размерность массива, flatten — наооборот
# # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#numpy.newaxis
# # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.flatten.html
#
# # Проверяем, действительно ли всё шкалировалось
# X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
#     X_std, y_std, test_size=0.3, random_state=0)
# print("результат шкалирования X_train")
# print("СКО",X_train_scaled.std())
# print("среднее",X_train_scaled.mean())
#
# #  Лассо регрессия
# print("Лассо регрессия")
# from sklearn.linear_model import Lasso
#
# lasso = Lasso(alpha=0.1)
# lasso.fit(X_train_scaled, y_train_scaled)
# y_train_pred = lasso.predict(X_train_scaled)
# y_test_pred = lasso.predict(X_test_scaled)
# print(lasso.coef_)
#
# print('ошибка СКО: {:.3f}, test: {:.3f}'.format(
#         mean_squared_error(y_train_scaled, y_train_pred),
#         mean_squared_error(y_test_scaled, y_test_pred)))
# print('коэффициент детерминации: {:.3f}, test: {:.3f}'.format(
#         r2_score(y_train_scaled, y_train_pred),
#         r2_score(y_test_scaled, y_test_pred)))
#
# # Ридж регрессия
# print("Ридж регрессия")
# from sklearn.linear_model import Ridge
#
# ridge = Ridge(alpha=0.1)
# ridge.fit(X_train_scaled, y_train_scaled)
# y_train_pred = ridge.predict(X_train_scaled)
# y_test_pred = ridge.predict(X_test_scaled)
# print(ridge.coef_)
#
# print('ошибка СКО: {:.3f}, test: {:.3f}'.format(
#         mean_squared_error(y_train_scaled, y_train_pred),
#         mean_squared_error(y_test_scaled, y_test_pred)))
# print('коэффициент детерминации: {:.3f}, test: {:.3f}'.format(
#         r2_score(y_train_scaled, y_train_pred),
#         r2_score(y_test_scaled, y_test_pred)))
#
# #  Регуляризация в ElasticNet
# print("ElasticNet")
# from sklearn.linear_model import ElasticNet
#
# en = ElasticNet(alpha=0.1, l1_ratio=0.5)
# en.fit(X_train_scaled, y_train_scaled)
# y_train_pred = en.predict(X_train_scaled)
# y_test_pred = en.predict(X_test_scaled)
# print(en.coef_)
#
# print('ошибка СКО: {:.3f}, test: {:.3f}'.format(
#         mean_squared_error(y_train_scaled, y_train_pred),
#         mean_squared_error(y_test_scaled, y_test_pred)))
# print('коэффициент детерминации: {:.3f}, test: {:.3f}'.format(
#         r2_score(y_train_scaled, y_train_pred),
#         r2_score(y_test_scaled, y_test_pred)))
#
#
# # Полиноминальная регрессия
# """
# Не всегда аппроксимация в виде прямой линии является наилучшим выходом.
# Иногда, стоит отказаться от предположения о наличии такой связи и воспользоваться полиноминальной
# регрессией.
# """
# print("Полиноминальная регрессия")
# from sklearn.preprocessing import PolynomialFeatures
# X = boston_df[['LSTAT']].values
# y = boston_df['MEDV'].values
#
# regr = LinearRegression()
#
# # создание полиномиальных объектов
# quadratic = PolynomialFeatures(degree=2)
# cubic = PolynomialFeatures(degree=3)
# X_quad = quadratic.fit_transform(X)
# X_cubic = cubic.fit_transform(X)
#
# # подходящие характеристики
# X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
#
# regr = regr.fit(X, y)
#
# y_lin_fit = regr.predict(X_fit)
# linear_r2 = r2_score(y, regr.predict(X))
#
# regr = regr.fit(X_quad, y)
# y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
# quadratic_r2 = r2_score(y, regr.predict(X_quad))
#
# regr = regr.fit(X_cubic, y)
# y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
# cubic_r2 = r2_score(y, regr.predict(X_cubic))
#
# # построение графиков
# plt.scatter(X, y, label='training points', color='lightgray')
#
# plt.plot(X_fit, y_lin_fit,
#          label='linear (d=1), $R^2={:.2f}$'.format(linear_r2),
#          color='blue',
#          lw=2,
#          linestyle=':')
#
# plt.plot(X_fit, y_quad_fit,
#          label='quadratic (d=2), $R^2={:.2f}$'.format(quadratic_r2),
#          color='red',
#          lw=2,
#          linestyle='-')
#
# plt.plot(X_fit, y_cubic_fit,
#          label='cubic (d=3), $R^2={:.2f}$'.format(cubic_r2),
#          color='green',
#          lw=2,
#          linestyle='--')
#
# plt.xlabel('% lower status of the population [LSTAT]')
# plt.ylabel('Price in $1000\'s [MEDV]')
# plt.legend(loc='upper right')
# pyplot.show()
#
# #  Множественная регрессия в Python
#
# y = [1,2,3,4,3,4,5,3,5,5,4,5,4,5,4,5,6,0,6,3,1,3,1]
# X = [[0,2,4,1,5,4,5,9,9,9,3,7,8,8,6,6,5,5,5,6,6,5,5],
#      [4,1,2,3,4,5,6,7,5,8,7,8,7,8,7,8,6,8,9,2,1,5,6],
#      [4,1,2,5,6,7,8,9,7,8,7,8,7,4,3,1,2,3,4,1,3,9,7]]
#
# def reg_m(y, x):
#     ones = np.ones(len(x[0]))
#     X = sm.add_constant(np.column_stack((x[0], ones)))
#     for ele in x[1:]:
#         X = sm.add_constant(np.column_stack((ele, X)))
#     results = sm.OLS(y, X).fit()
#     return results
#
# print(reg_m(y, X).summary())
#
# #  Аналогичный результат можно получить иначе
# X = np.transpose(X) # transpose so input vectors
# X = np.c_[X, np.ones(X.shape[0])]  # add bias term
# linreg = np.linalg.lstsq(X, y, rcond=None)[0]
# print(linreg)
#
# ========================================================
# #  Алгоритмы K-ближайших соседей и K-средних
# """
# Задачи классификации — это ситуации, когда у вас есть набор данных, и вы хотите классифицировать
# наблюдения из этого набора в определенную категорию.
#
# Пример задачи классификации — спам-фильтр для электронной почты. Gmail использует методы машинного
# обучения с учителем, чтобы автоматически помещать электронные письма в папку для спама в зависимости
# от их содержания, темы и других характеристик.
#
# Существуют две модели машинного обучения, которые выполняют большую часть работы, когда дело доходит
# до задач классификации:
# * Метод K-ближайших соседей
# * Метод К-средних
# """
# print("Модели K-ближайших соседей")
# # Модели K-ближайших соседей
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
# raw_data = pd.read_csv('classified_data.csv')
# """
# Таблица начинается с безымянного столбца, значения которого равны номерам строк DataFrame.
# Мы можем исправить это, немного изменив команду, добавив index_col = 0 после файла
# """
# raw_data = pd.read_csv('classified_data.csv', index_col = 0)
#
# #  выведим список имен столбцов
# print(raw_data.columns)
#
# """
# Стандартизация датасета
# Поскольку алгоритм K-ближайших соседей делает прогнозы относительно точки данных (семпла),
# используя наиболее близкие к ней наблюдения, существующий масштаб показателей в датасете имеет большое
# значение. Из-за этого обычно стандартизируются наборы данных, что означает корректировку каждого значения
# x так, чтобы они находились примерно в одном диапазоне.
#
# Это позволяет сделать библиотека scikit-learn.
# """
# from sklearn.preprocessing import StandardScaler
#
# """
# Этот класс во многом похож на классы LinearRegression и LogisticRegression.
# Нам нужно создать экземпляр StandardScaler, а затем использовать этот объект для преобразования наших данных.
# """
#
# scaler = StandardScaler()
# # обучаем scaler на нашем датасете, используя метод fit
# scaler.fit(raw_data.drop('TARGET CLASS', axis=1))
# #  применим метод transform для стандартизации всех признаков, чтобы они имели примерно одинаковый масштаб
# scaled_features = scaler.transform(raw_data.drop('TARGET CLASS', axis=1))
#
# """
# В качестве результата получили массив NumPy со всеми точками данных из датасета, но нам желательно
# преобразовать его в формат DataFrame библиотеки pandas. Для этого обернем переменную scaled_features
# в метод pd.DataFrame и назначим этот DataFrame новой переменной scaled_data с соответствующим аргументом
# для указания имен столбцов
# """
#
# scaled_data = pd.DataFrame(scaled_features, columns = raw_data.drop('TARGET CLASS', axis=1).columns)
#
# """
# # Деление датасета на обучающие и тестовые данные
# Используем функцию train_test_split библиотеки scikit-learn в сочетании с распаковкой списка для создания
# обучающих и тестовых датасетов из нашего набора. Для этого нужно импортировать train_test_split из модуля
# model_validation библиотеки scikit-learn.
# """
#
# from sklearn.model_selection import train_test_split
#
# """
# Затем необходимо указать значения x и y, которые будут переданы в функцию train_test_split.
# Значения x представляют собой DataFrame scaled_data, который мы создали.
# Значения y хранятся в столбце "TARGET CLASS" нашей исходной таблицы raw_data.
# """
# x = scaled_data
# y = raw_data['TARGET CLASS']
#
# """
# Запускаем функцию train_test_split, используя эти два аргумента и test_size, равный 30%,
# """
# x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)
#
# #  Обучение модели K-ближайших соседей
# # импорт KNeighborsClassifier из scikit-learn
#
# from sklearn.neighbors import KNeighborsClassifier
#
# """
# создадим экземпляр класса KNeighborsClassifier и назначим его переменной model.
# Для этого требуется передать параметр n_neighbors, который равен выбранному вами значению K
# алгоритма K-ближайших соседей. Для начала укажем n_neighbors = 1.
# """
#
# model = KNeighborsClassifier(n_neighbors = 1)
# #  обучим модель, используя метод fit и переменные x_training_data и y_training_data
# model.fit(x_training_data, y_training_data)
#
# """
# Делаем предсказания с помощью алгоритма K-ближайших соседей
# Способ получения прогнозов на основе алгоритма K-ближайших соседей такой же, как и у моделей линейной и
# логистической регрессий: для предсказания достаточно вызвать метод predict, передав в него переменную
# x_test_data.
# """
#
# predictions = model.predict(x_test_data)
#
# """
# Оценка точности модели
# scikit-learn поставляется со встроенными функциями, которые упрощают измерение эффективности
# классификационных моделей машинного обучения.
# Для этого импортируем в отчет две функции classification_report и confusion_matrix
# """
#
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# #  создание отчета
# print("поддержка, оценка, точность отзыв, точность, среднее, взвешенное среднее")
# print(classification_report(y_test_data, predictions))
# # сгенерируем матрицу ошибок
# print("матрица ошибок")
# print(confusion_matrix(y_test_data, predictions))
#
# """
# Метода «Локтя»
# Метод «локтя» позволяет выбрать оптимальное значение K для алгоритма K-ближайших соседей.
# Он включает в себя итерацию по различным значениям K и выбор значения с наименьшей частотой ошибок
# при применении к тестовым данным.
# Для начала создается пустой список error_rates и перебираются различные значения K, добавляя
# их частоту ошибок в этот список.
# """
#
# error_rates = []
#
# """
# Создаем цикл, который перебирает различные значения K, которые хотим протестировать,
# на каждой итерации выполняет следующее:
# * Создает новый экземпляр класса KNeighborsClassifier из scikit-learn.
# * Тренирует эту модель, используя наши обучающие данные.
# * Делает прогнозы на основе наших тестовых данных.
# * Вычисляет долю неверных предсказаний (чем она ниже, тем точнее наша модель).
# """
# for i in np.arange(1, 101):
#     new_model = KNeighborsClassifier(n_neighbors = i)
#     new_model.fit(x_training_data, y_training_data)
#     new_predictions = new_model.predict(x_test_data)
#     error_rates.append(np.mean(new_predictions != y_test_data))
# plt.plot(error_rates)
# pyplot.show()
# """
# Как видно из графика, минимальная частота ошибок при значении K достигается при 35.
# Это означает, что 37 является подходящим выбором для K, который сочетает в себе простоту и точность
# предсказаний.
# """
# print("метод K-средних")
#
# """
# Модели кластеризации методом K-средних
#
# Алгоритм кластеризации K-средних является моделью машинного обучения без учителя.
# Он позволяет создавать группы точек данных со схожими количественными характеристиками в датасете.
# Это полезно для решения таких задач, как формирование клиентских сегментов или определение городских
# районов с высоким уровнем преступности.
# Применяем модель кластеризации K-средних для получения предсказаний обычно используют алгоритмы
# кластеризации, чтобы делать два типа прогнозов:
# * К какому кластеру принадлежит каждая точка данных.
# * Где находится центр каждого кластера.
#
# Используемый датасет будет создан с помощью scikit-learn.
# Необходимо импортировать функцию make_blobs из scikit-learn, чтобы сгенерировать необходимые данные.
# """
# from sklearn.datasets import make_blobs
#
# """
# Будем использовать функцию make_blobs, чтобы получить фиктивные данные из 200 семплов,
# который имеет 2 показателя и 4 кластерных центра.
# Стандартное отклонение для каждого кластера будет равно 1,8.
# """
# raw_data = make_blobs(
#     n_samples = 200,
#     n_features = 2,
#     centers = 4,
#     cluster_std = 1.8
# )
# print(raw_data)
# """
# raw_data - кортеж, первым его элементом является массив NumPy с 200 наблюдениями.
# Каждое наблюдение содержит 2 признака
# """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
# plt.scatter(raw_data[0][:,0], raw_data[0][:,1], c=raw_data[1]);
# pyplot.show()
#
# """"
# Создание и обучение модели кластеризации K-средних
# Импортируем соответствующий класс из scikit-learn.
# """
# from sklearn.cluster import KMeans
#
# #  создадим экземпляр класса KMeans с параметром n_clusters=4 и присвоим его переменной model
# model = KMeans(n_clusters=4)
#
# # обучим модель, вызвав на ней метод fit и передав первый элемент кортежа raw_data
# model.fit(raw_data[0])
#
# # определение какому кластеру принадлежит каждая точка данных
# model.labels_
# # определение центра каждого кластера
# model.cluster_centers_
# print("центры кластеров",model.cluster_centers_)
# # Визуализация точности предсказаний модели
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
# ax1.se_title('Наши предсказания')
# ax1.scatter(raw_data[0][:,0], raw_data[0][:,1],c=model.labels_)
# ax2.set_title('Реальные значения')
# ax2.scatter(raw_data[0][:,0], raw_data[0][:,1],c=raw_data[1]);
# pyplot.show()
#
# ============================================================
# Ранговая корреляция
# генерация связанных переменных
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot
# генератор начальных случайных чисел
# вычисление случайных данных
data1 = rand(1000) * 20
data2 = data1 + (rand(1000) * 10)
seed(1)
print(data1)
print(data2)
# построение графика по сгенирированным данным
pyplot.scatter(data1, data2)
pyplot.show()