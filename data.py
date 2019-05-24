import warnings

warnings.filterwarnings("ignore")
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


def in_data():
    df = pd.read_excel('...data/try34.xlsx')
    dete = df.loc[df['disease'] == 'typhoid']
    dete['  period'].min(), dete['  period'].max()

    # deleting unuseful columns
    cols = ['disease', 'no. of medicines', 'no. of specialist']
    dete.drop(cols, axis=1, inplace=True)

    # sorting date_time and indexing
    dete = dete.sort_values('  period')
    dete.isnull().sum()
    dete = dete.groupby('  period')['no. of patients'].sum().reset_index()
    dete = dete.set_index('  period')
    dete.index

    y = dete['no. of patients'].resample('MS').mean()
    # print data
    print('Showing data that user inputed in graphical form.')
    y.plot(figsize=(15, 6))
    plt.show()

    from pylab import rcParams
    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
    fig = decomposition.plot()
    print('decomposition model in three different categores as trend, seasonal and observed')
    plt.show()

    p = d = q = range(0, 5)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=(4, 2, 1),
                                    seasonal_order=(0, 0, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()

    # sumary
    print('summary of an model', results.summary().tables[1])

    results.plot_diagnostics(figsize=(16, 8))
    plt.show()

    print('The predicted model as shown as below.')

    plt.subplot(2, 1, 1)
    pred = results.forecast(steps=5)
    ax = y['2014':].plot(label='observed')
    pred.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.set_xlabel('Date')
    ax.set_ylabel('no. of patients')
    plt.title('Predictive Analysis on typhoid data')
    plt.legend()
    plt.grid('True')

    data = ['Jan', 'Feb', 'March', 'April', 'May']
    plt.subplot(2, 1, 2)
    plt.pie(pred, labels=data, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.legend(pred)
    plt.axis('equal')
    plt.tight_layout()

    plt.title('Monthly Predictive Analysis On Disease : typhoid')
    plt.savefig('yes.png')
    plt.show()




    final = y['2014':].append(pred)
    return final, y

in_data()