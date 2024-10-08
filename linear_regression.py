from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/danielhernandez/Downloads/Watershed_Characteristics.csv')

print(df.info())

y2 = df['Annual Avg. River Flow']
x1 = df['Aridity Index']

def linear_regression_train(x, y):
    slope, intercept, r, p, std_error = stats.linregress(x,y)
    linear_regression = slope * x + intercept

    plt.scatter(x,y)
    plt.plot(x,linear_regression)
    plt.xlabel(f'{x.name} ', size=15)
    plt.ylabel(f'{y.name} (mm)', size=15)
    plt.title(f'{x.name} vs {y.name}', size=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(), print(), print(f'{x.name} vs {y.name}')
    print('-------------------')
    print(f'{x.name} vs {y.name} slope:     ', slope)
    print(f'{x.name} vs {y.name} intercept: ', intercept)
    print(f'{x.name} vs {y.name} r:         ', r)
    print(f'{x.name} vs {y.name} r^2:       ', r*r)
    print(f'{x.name} vs {y.name} std_error: ', std_error)

linear_regression_train(x1,y2)