[TCC_Finances - Copia.md](https://github.com/user-attachments/files/23291156/TCC_Finances.-.Copia.md)
# **Comparação de Investimentos Utilizando Modelos de Machine Learning para Previsão de Retornos e Análise de Risco**


Este trabalho tem como objetivo desenvolver um modelo comparador de investimentos que utilize técnicas de Machine Learning para prever os retornos e analisar o risco de diferentes opções de investimento, incluindo ações, fundos de investimento e ETFs. Através da coleta e análise de dados históricos de preços, volumes, e indicadores financeiros, será possível treinar modelos preditivos e avaliar a performance de cada investimento, auxiliando investidores na tomada de decisões mais informadas.

Objetivo Geral:
Comparar diferentes opções de investimento (ações, fundos de investimento e ETFs) com base em previsões de retorno e análise de risco, utilizando modelos de Machine Learning.

Objetivos Específicos:


*   Coletar e preparar dados históricos de ações da B3, fundos de investimento da ANBIMA e ETFs.
*   Realizar análises exploratórias para entender a distribuição e padrões nos dados.
*   Desenvolver modelos preditivos utilizando algoritmos de Machine Learning para prever os retornos dos diferentes investimentos.
*   Avaliar o desempenho dos modelos usando métricas de erro (RMSE, MAE).
*   Comparar os investimentos com base nas previsões de retorno e análise de risco, incluindo volatilidade e outros indicadores financeiros.
*   Fornecer recomendações sobre quais investimentos apresentam melhor retorno ajustado pelo risco.








# 1 - Organizar Ferramentas:



## 1.1 Importar Pacotes


```python
# Pacotes

import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from scipy.stats import shapiro
from scipy.optimize import minimize

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

import pmdarima as pm
from pmdarima import auto_arima

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error , r2_score
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
```

## 1.2 Funções


```python
def std_deviation(weights, cov_matrix):
    '''
    Calcula o desvio padrão do portfolio.
    Primeiro calcula a variância do portfolio, medida de risco associada a portfolios. Representa a volatilidade combinada dos ativos no portfolio,
    levando em consideração volatilidades individuais e correlações entre eles.
    '''
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

```


```python
def expected_returns(weights, log_returns):
    '''
    Calcula o retorno esperado do portfolio. Baseado em retornos históricos
    '''
    return np.sum(log_returns.mean() * weights) * 252
```


```python
def sharpe_ratio (weights, log_returns, cov_matrix, risk_free_rate):
    '''
    Calcula o ratio de Sharpe do portfolio.
    '''
    return (expected_returns(weights, log_returns) - risk_free_rate) / std_deviation(weights, cov_matrix)
```


```python
def negative_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    '''
    Calcula o ratio de Sharpe negativo. (Maximização)
    '''
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)
```


```python
def decompor_serie_temporal(serie, model='additive', period=252, plot=True):
    """
    Decompõe uma série temporal em seus componentes: tendência, sazonalidade e resíduos.

    Parameters:
    - serie (pd.Series): A série temporal a ser decomposta.
    - model (str): Tipo de modelo para decomposição ('additive' ou 'multiplicative'). Padrão é 'additive'.
    - period (int): O período da sazonalidade. Padrão é 252 (aproximadamente um ano de dados diários).
    - plot (bool): Se True, plota os componentes da decomposição. Padrão é True.

    Returns:
    - decomposition (DecomposeResult): O resultado da decomposição contendo os componentes observados, tendência, sazonalidade e resíduos.
    """
    # Realizar a decomposição sazonal
    decomposition = seasonal_decompose(serie, model=model, period=period)

    # Plotar os resultados, se necessário
    if plot:
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        decomposition.observed.plot(ax=axes[0], title='Observado')
        decomposition.trend.plot(ax=axes[1], title='Tendência')
        decomposition.seasonal.plot(ax=axes[2], title='Sazonalidade')
        decomposition.resid.plot(ax=axes[3], title='Resíduos')
        plt.tight_layout()
        plt.show()

    return decomposition
```


```python

def teste_normalidade(residuos, ticker):
    """
    Testa a normalidade dos resíduos usando o Teste de Shapiro-Wilk.

    Parameters:
    - residuos (pd.Series): Resíduos da decomposição.
    - ticker (str): Nome do ativo ou ticker.

    Returns:
    - None
    """
    statistic, p_value = shapiro(residuos.dropna())
    print(f'\nTeste de Shapiro-Wilk para {ticker}: Estatística={statistic:.3f}, p-valor={p_value:.3f}')
    if p_value > 0.05:
        print('Provavelmente normal')
    else:
        print('Provavelmente não normal')

```


```python

def teste_estacionariedade(serie_temporal, ticker):
    """
    Testa a estacionariedade da série temporal usando o Teste de Dickey-Fuller Aumentado.

    Parameters:
    - serie_temporal (pd.DataFrame): Série temporal a ser testada.
    - ticker (str): Nome do ativo ou ticker.

    Returns:
    - None
    """
    result = adfuller(serie_temporal[ticker].dropna())
    print(f'Teste de Dickey-Fuller Aumentado para {ticker}:')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] <= 0.05:
        print('Série provavelmente estacionária\n')
    else:
        print('Série provavelmente não estacionária\n')

```


```python


def analise_autocorrelacao(residuos, ticker):
    """
    Plota a autocorrelação e autocorrelação parcial dos resíduos.

    Parameters:
    - residuos (pd.Series): Resíduos da decomposição.
    - ticker (str): Nome do ativo ou ticker.

    Returns:
    - None
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(residuos.dropna(), ax=axes[0])
    plot_pacf(residuos.dropna(), ax=axes[1])
    plt.suptitle(f'Autocorrelação e Autocorrelação Parcial para {ticker}')
    plt.tight_layout()
    plt.show()

```


```python
def plot_cumulative_returns(data):
  """
  Plota os retornos acumulados de um dataframe contendo dados de ações.

  Args:
    data: DataFrame contendo os dados de retornos de ações, com uma coluna 'data'.
  """

  # Calcula os retornos acumulados
  cumulative_returns = (1 + data).cumprod() - 1

  return cumulative_returns

```


```python
def create_df(df, steps=1):
    '''
    Cria dados de treino e teste para um modelo de machine learning. Em formato de Série Temporal
    '''
    dataX , dataY = [], []
    for i in range(len(df)-steps-1):
        a = df[i:(i+steps), 0]
        dataX.append(a)
        dataY.append(df[i + steps, 0])
    return np.array(dataX), np.array(dataY)
```


```python
# Função para Regressão Linear
def regressao_linear(df, coluna , best_window = 252):
    """
    Executa uma regressão linear para prever valores com base em uma coluna específica.
    
    Args:
        df (DataFrame): DataFrame contendo os dados históricos.
        coluna (str): Nome da coluna de interesse para a regressão.

    Returns:
        dict: Contém o modelo ajustado, métricas (MSE, MAE, R²) e previsões futuras.
    """
    # Extraindo as variáveis independentes e dependentes
    X = df['Date_ordinal'].values.reshape(-1, 1)  # Data em formato ordinal como variável independente
    y = df[coluna].values  # Coluna como variável dependente

    # Ajustando o modelo de regressão linear
    modelo = LinearRegression()
    modelo.fit(X, y)

    # Fazendo previsões nos dados históricos
    previsoes_historicas = modelo.predict(X)

    # Gerando previsões futuras
    dias_previsao = best_window  # Número de dias úteis no futuro
    ultima_data = df.index[-1]
    datas_futuras = pd.date_range(start=ultima_data, periods=dias_previsao + 1, freq='B')[1:]
    datas_ordinais_futuras = datas_futuras.map(lambda x: x.toordinal()).values.reshape(-1, 1)
    previsoes_futuras = modelo.predict(datas_ordinais_futuras)

    # Calculando métricas
    mse = mean_squared_error(y, previsoes_historicas)
    mae = mean_absolute_error(y, previsoes_historicas)
    r2 = r2_score(y, previsoes_historicas)

    # Retornando resultados
    return {
        'modelo': modelo,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'previsoes_historicas': previsoes_historicas,
        'previsoes_futuras': previsoes_futuras,
        'datas_futuras': datas_futuras
    }


```

# 2 - Definir os Ativos e o Período de Tempo


```python
# Definir ativos do portfolio desejado
portfolio_value = 100*1000
benchmark = ['^BVSP']
portfolio = ['PETR4.SA','SBSP3.SA','RENT3.SA','ITUB4.SA','VALE3.SA'] # Selecione os tickers desejados
weights = [0.2, 0.2, 0.2, 0.2, 0.2]

# Definir datas e períodos
years = 3

end_date = datetime.today()
start_date = end_date - timedelta(days=365*years)
```

# 3 - Coleta dos Dados

## 3.1 Histórico de Ativos


```python
# Criar um dataframe vazio para armazenar o preços ajustados de fechamento dos ativos

df_adj = pd.DataFrame()

# Baixar os dados de fechamento de cada ticker

for tickers in portfolio:
  df_adj[tickers] = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
  df_adj.fillna(0 , inplace=True)

# Mostrar o dataframe

df_adj
```

    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    [*********************100%%**********************]  1 of 1 completed
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PETR4.SA</th>
      <th>SBSP3.SA</th>
      <th>RENT3.SA</th>
      <th>ITUB4.SA</th>
      <th>VALE3.SA</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-12-28</th>
      <td>11.357904</td>
      <td>37.474430</td>
      <td>48.797234</td>
      <td>18.368296</td>
      <td>60.518539</td>
    </tr>
    <tr>
      <th>2021-12-29</th>
      <td>11.263190</td>
      <td>37.296600</td>
      <td>47.475925</td>
      <td>18.214378</td>
      <td>60.675632</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>11.227673</td>
      <td>37.605461</td>
      <td>48.296059</td>
      <td>17.915079</td>
      <td>61.233299</td>
    </tr>
    <tr>
      <th>2022-01-03</th>
      <td>11.480244</td>
      <td>37.390194</td>
      <td>46.373322</td>
      <td>18.409466</td>
      <td>61.264713</td>
    </tr>
    <tr>
      <th>2022-01-04</th>
      <td>11.523655</td>
      <td>36.360680</td>
      <td>46.637577</td>
      <td>18.931540</td>
      <td>60.542107</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2024-12-18</th>
      <td>35.965580</td>
      <td>87.900002</td>
      <td>29.930000</td>
      <td>30.969999</td>
      <td>54.810001</td>
    </tr>
    <tr>
      <th>2024-12-19</th>
      <td>35.820984</td>
      <td>87.190002</td>
      <td>32.549999</td>
      <td>31.139999</td>
      <td>53.770000</td>
    </tr>
    <tr>
      <th>2024-12-20</th>
      <td>35.522152</td>
      <td>86.620003</td>
      <td>33.680000</td>
      <td>31.500000</td>
      <td>54.619999</td>
    </tr>
    <tr>
      <th>2024-12-23</th>
      <td>35.531796</td>
      <td>88.400002</td>
      <td>32.880001</td>
      <td>30.889999</td>
      <td>54.849998</td>
    </tr>
    <tr>
      <th>2024-12-26</th>
      <td>35.770000</td>
      <td>88.559998</td>
      <td>32.529999</td>
      <td>31.090000</td>
      <td>55.009998</td>
    </tr>
  </tbody>
</table>
<p>750 rows × 5 columns</p>
</div>



## 3.2 Taxa Básica (SELIC)


```python
# URL para acessar a taxa SELIC diária do Banco Central
url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=json"

url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=json&dataInicial={}&dataFinal={}".format(
    start_date.strftime('%d/%m/%Y'), end_date.strftime('%d/%m/%Y')
)

# Solicitar os dados
response = requests.get(url)
data = response.json()

# Converter para DataFrame
selic = pd.DataFrame(data)

# Converter a coluna data para datetime e valor para float
selic['data'] = pd.to_datetime(selic['data'], format='%d/%m/%Y')
selic['valor'] = selic['valor'].astype(float)
selic = selic.set_index('data')


# Mostrar os primeiros registros
selic.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>valor</th>
    </tr>
    <tr>
      <th>data</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-12-19</th>
      <td>0.045513</td>
    </tr>
    <tr>
      <th>2024-12-20</th>
      <td>0.045513</td>
    </tr>
    <tr>
      <th>2024-12-23</th>
      <td>0.045513</td>
    </tr>
    <tr>
      <th>2024-12-24</th>
      <td>0.045513</td>
    </tr>
    <tr>
      <th>2024-12-26</th>
      <td>0.045513</td>
    </tr>
  </tbody>
</table>
</div>



## 3.3 CDI


```python
# Baixar dados do CDI diário
url_cdi = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json&dataInicial={}&dataFinal={}".format(
    start_date.strftime('%d/%m/%Y'), end_date.strftime('%d/%m/%Y')
)
response_cdi = requests.get(url_cdi)
data_cdi = response_cdi.json()

# Converter para DataFrame
cdi = pd.DataFrame(data_cdi)
cdi['data'] = pd.to_datetime(cdi['data'], format='%d/%m/%Y')
cdi['valor'] = cdi['valor'].astype(float) / 100  # Converter de porcentagem para decimal
cdi = cdi.set_index('data')
cdi = cdi.rename(columns={'valor': 'cdi'})

cdi.tail()

# Estabelecendo a taxa livre de risco

risk_free_rate = cdi['cdi'].iloc[-1]
risk_free_rate
```




    0.00045513



## 3.4 Indice Ibovespa


```python
# Baixar dados históricos do IBOV (Índice Bovespa)
ibov = pd.DataFrame()
ibov['IBOV'] = yf.download('^BVSP', start=start_date, end=end_date)['Adj Close']

ibov.tail()
```

    [*********************100%%**********************]  1 of 1 completed
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IBOV</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-12-18</th>
      <td>120772.0</td>
    </tr>
    <tr>
      <th>2024-12-19</th>
      <td>121188.0</td>
    </tr>
    <tr>
      <th>2024-12-20</th>
      <td>122102.0</td>
    </tr>
    <tr>
      <th>2024-12-23</th>
      <td>120767.0</td>
    </tr>
    <tr>
      <th>2024-12-26</th>
      <td>121077.5</td>
    </tr>
  </tbody>
</table>
</div>



# 4 - Preparação dos Dados

## 4.1 Calculo dos Retornos em Log-Normal


```python
# Calculo dos retornos de cada ticker em log-normal

log_returns = np.log(df_adj/df_adj.shift(1))

# Apagar dados ausentes

log_returns.dropna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PETR4.SA</th>
      <th>SBSP3.SA</th>
      <th>RENT3.SA</th>
      <th>ITUB4.SA</th>
      <th>VALE3.SA</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-12-29</th>
      <td>-0.008374</td>
      <td>-0.004757</td>
      <td>-0.027451</td>
      <td>-0.008415</td>
      <td>0.002592</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>-0.003158</td>
      <td>0.008247</td>
      <td>0.017127</td>
      <td>-0.016569</td>
      <td>0.009149</td>
    </tr>
    <tr>
      <th>2022-01-03</th>
      <td>0.022246</td>
      <td>-0.005741</td>
      <td>-0.040626</td>
      <td>0.027222</td>
      <td>0.000513</td>
    </tr>
    <tr>
      <th>2022-01-04</th>
      <td>0.003774</td>
      <td>-0.027921</td>
      <td>0.005682</td>
      <td>0.027964</td>
      <td>-0.011865</td>
    </tr>
    <tr>
      <th>2022-01-05</th>
      <td>-0.039467</td>
      <td>-0.043937</td>
      <td>-0.029344</td>
      <td>-0.019170</td>
      <td>0.009426</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2024-12-18</th>
      <td>-0.026188</td>
      <td>-0.030144</td>
      <td>-0.027851</td>
      <td>-0.028960</td>
      <td>-0.023441</td>
    </tr>
    <tr>
      <th>2024-12-19</th>
      <td>-0.004029</td>
      <td>-0.008110</td>
      <td>0.083916</td>
      <td>0.005474</td>
      <td>-0.019157</td>
    </tr>
    <tr>
      <th>2024-12-20</th>
      <td>-0.008377</td>
      <td>-0.006559</td>
      <td>0.034127</td>
      <td>0.011494</td>
      <td>0.015684</td>
    </tr>
    <tr>
      <th>2024-12-23</th>
      <td>0.000271</td>
      <td>0.020341</td>
      <td>-0.024040</td>
      <td>-0.019555</td>
      <td>0.004202</td>
    </tr>
    <tr>
      <th>2024-12-26</th>
      <td>0.006682</td>
      <td>0.001808</td>
      <td>-0.010702</td>
      <td>0.006454</td>
      <td>0.002913</td>
    </tr>
  </tbody>
</table>
<p>749 rows × 5 columns</p>
</div>



## 4.2 Cálculo dos Retornos do Índice IBOV Diário


```python
log_ibov = np.log(ibov/ibov.shift(1)).dropna()
log_ibov
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IBOV</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-12-29</th>
      <td>-0.007245</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>0.006844</td>
    </tr>
    <tr>
      <th>2022-01-03</th>
      <td>-0.008623</td>
    </tr>
    <tr>
      <th>2022-01-04</th>
      <td>-0.003934</td>
    </tr>
    <tr>
      <th>2022-01-05</th>
      <td>-0.024527</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2024-12-18</th>
      <td>-0.031990</td>
    </tr>
    <tr>
      <th>2024-12-19</th>
      <td>0.003439</td>
    </tr>
    <tr>
      <th>2024-12-20</th>
      <td>0.007514</td>
    </tr>
    <tr>
      <th>2024-12-23</th>
      <td>-0.010994</td>
    </tr>
    <tr>
      <th>2024-12-26</th>
      <td>0.002568</td>
    </tr>
  </tbody>
</table>
<p>749 rows × 1 columns</p>
</div>



## 4.3 Cálculo dos Retornos do CDI


```python
log_cdi = np.log(cdi/cdi.shift(1)).dropna()
log_cdi

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cdi</th>
    </tr>
    <tr>
      <th>data</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-12-28</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-12-29</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-12-31</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-01-03</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2024-12-18</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2024-12-19</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2024-12-20</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2024-12-23</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2024-12-24</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>753 rows × 1 columns</p>
</div>



## 4.4 Cálculo dos Retornos Históricos do Portfolio


```python
historical_returns = (log_returns * weights).sum(axis=1)
historical_returns = historical_returns.rename('returns')
historical_returns

```




    Date
    2021-12-28    0.000000
    2021-12-29   -0.009281
    2021-12-30    0.002959
    2022-01-03    0.000723
    2022-01-04   -0.000473
                    ...   
    2024-12-18   -0.027317
    2024-12-19    0.011619
    2024-12-20    0.009274
    2024-12-23   -0.003756
    2024-12-26    0.001431
    Name: returns, Length: 750, dtype: float64



# 5 - Análise Exploratória dos Dados

### Preço


```python
# Plotar os preços de fechamento ajustados de cada ativo
for ticker in portfolio:
    plt.figure(figsize=(12, 6))
    plt.plot(df_adj.index, df_adj[ticker], label=ticker)

    # Adicionar benchmarks
    plt.plot(ibov.index, ibov['IBOV'] / ibov['IBOV'].iloc[0] * df_adj[ticker].iloc[0], label='IBOV', linestyle=':')

    plt.title('Preço de Fechamento Ajustado - {}'.format(ticker))
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.legend()
    plt.grid(True)
    plt.show()

```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_40_0.png)
    



    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_40_1.png)
    



    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_40_2.png)
    



    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_40_3.png)
    



    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_40_4.png)
    



```python

# Plotar os preços de fechamento ajustados de cada ativo
plt.figure(figsize=(12, 6))

for ticker in portfolio:
    plt.plot(df_adj.index, df_adj[ticker], label=ticker)

# Adicionar o IBOV em destaque
plt.plot(ibov.index, ibov['IBOV'] / ibov['IBOV'].iloc[0] * df_adj[portfolio[0]].iloc[0], label='IBOV', linewidth=2, color='black')
plt.title('Preço de Fechamento Ajustado - Portfolio vs IBOV')
plt.xlabel('Data')
plt.ylabel('Preço')
plt.legend()
plt.grid(True)
plt.show()

```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_41_0.png)
    


### Retornos


```python
# Plotar os retornos diários de cada ativo
for ticker in log_returns.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(log_returns.index, log_returns[ticker].cumsum(), label=ticker)

    # Adicionar benchmarks
    plt.plot(cdi.index, cdi['cdi'].cumsum(), label='CDI', linestyle='--')
    plt.plot(log_ibov.index, log_ibov['IBOV'].cumsum(), label='IBOV', linestyle='-.')
    plt.axhline(0,color='k')


    plt.title('Retorno Acumulado - {}'.format(ticker))
    plt.xlabel('Data')
    plt.ylabel('Retorno Acumulado')
    plt.legend()
    plt.grid(True)
    plt.show()

```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_43_0.png)
    



    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_43_1.png)
    



    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_43_2.png)
    



    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_43_3.png)
    



    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_43_4.png)
    



```python
# Plota grãfico unificado
plt.figure(figsize=(12, 6))

# Plotar os retornos diários de cada ativo
for ticker in log_returns.columns:
    plt.plot(log_returns.index, log_returns[ticker].cumsum(), label=ticker)

# Adicionar benchmarks em destaque
plt.plot(cdi.index, cdi['cdi'].cumsum(), label='CDI', linestyle='--', linewidth=2, color='brown')
plt.plot(log_ibov.index, log_ibov['IBOV'].cumsum(), label='IBOV', linestyle='-.', linewidth=2, color='black')
plt.axhline(0,color='k')


plt.title('Retorno Acumulado - Portfolio vs Benchmarks')
plt.xlabel('Data')
plt.ylabel('Retorno Acumulado')
plt.legend()
plt.grid(True)
plt.show()

```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_44_0.png)
    


### Análise Temporal dos Ativos


```python
# Análise de todos os tickers do Portfolio

for ticker in log_returns:

    print("\n ---------  RESULTADOS ---------- \n")
    print(f"{ticker}\n\n")
    # Decomposição da série temporal
    decomposition = seasonal_decompose(df_adj[ticker], model='additive', period=252)  # Considerando um ano com 252 dias úteis

    # Plotar a decomposição
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    decomposition.observed.plot(ax=axes[0], title='Observado')
    decomposition.trend.plot(ax=axes[1], title='Tendência')
    decomposition.seasonal.plot(ax=axes[2], title='Sazonalidade')
    decomposition.resid.plot(ax=axes[3], title='Resíduos')
    plt.tight_layout()
    plt.show()

    # Teste de normalidade dos resíduos (Shapiro-Wilk)
    from scipy.stats import shapiro
    statistic, p_value = shapiro(decomposition.resid.dropna())
    print(f'\nTeste de Shapiro-Wilk para {ticker}: Estatística={statistic:.3f}, p-valor={p_value:.3f}')
    if p_value > 0.05:
        print('Provavelmente normal')
    else:
        print('Provavelmente não normal')

    # Teste de estacionariedade (Dickey-Fuller Aumentado)
    result = adfuller(df_adj[ticker])
    print(f'Teste de Dickey-Fuller Aumentado para {ticker}:')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] <= 0.05:
        print('Série provavelmente estacionária\n')
    else:
        print('Série provavelmente não estacionária\n')

    # Autocorrelação e Autocorrelação Parcial dos resíduos
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(decomposition.resid.dropna(), ax=axes[0])
    plot_pacf(decomposition.resid.dropna(), ax=axes[1])
    plt.tight_layout()
    plt.show()

```

    
     ---------  RESULTADOS ---------- 
    
    PETR4.SA
    
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_46_1.png)
    


    
    Teste de Shapiro-Wilk para PETR4.SA: Estatística=0.986, p-valor=0.000
    Provavelmente não normal
    Teste de Dickey-Fuller Aumentado para PETR4.SA:
    ADF Statistic: -0.727890
    p-value: 0.839393
    Critical Values:
    	1%: -3.439
    	5%: -2.865
    	10%: -2.569
    Série provavelmente não estacionária
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_46_3.png)
    


    
     ---------  RESULTADOS ---------- 
    
    SBSP3.SA
    
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_46_5.png)
    


    
    Teste de Shapiro-Wilk para SBSP3.SA: Estatística=0.990, p-valor=0.003
    Provavelmente não normal
    Teste de Dickey-Fuller Aumentado para SBSP3.SA:
    ADF Statistic: -0.725344
    p-value: 0.840075
    Critical Values:
    	1%: -3.439
    	5%: -2.865
    	10%: -2.569
    Série provavelmente não estacionária
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_46_7.png)
    


    
     ---------  RESULTADOS ---------- 
    
    RENT3.SA
    
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_46_9.png)
    


    
    Teste de Shapiro-Wilk para RENT3.SA: Estatística=0.974, p-valor=0.000
    Provavelmente não normal
    Teste de Dickey-Fuller Aumentado para RENT3.SA:
    ADF Statistic: -1.224633
    p-value: 0.662911
    Critical Values:
    	1%: -3.439
    	5%: -2.865
    	10%: -2.569
    Série provavelmente não estacionária
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_46_11.png)
    


    
     ---------  RESULTADOS ---------- 
    
    ITUB4.SA
    
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_46_13.png)
    


    
    Teste de Shapiro-Wilk para ITUB4.SA: Estatística=0.978, p-valor=0.000
    Provavelmente não normal
    Teste de Dickey-Fuller Aumentado para ITUB4.SA:
    ADF Statistic: -1.482946
    p-value: 0.541930
    Critical Values:
    	1%: -3.439
    	5%: -2.865
    	10%: -2.569
    Série provavelmente não estacionária
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_46_15.png)
    


    
     ---------  RESULTADOS ---------- 
    
    VALE3.SA
    
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_46_17.png)
    


    
    Teste de Shapiro-Wilk para VALE3.SA: Estatística=0.985, p-valor=0.000
    Provavelmente não normal
    Teste de Dickey-Fuller Aumentado para VALE3.SA:
    ADF Statistic: -2.292872
    p-value: 0.174277
    Critical Values:
    	1%: -3.439
    	5%: -2.865
    	10%: -2.569
    Série provavelmente não estacionária
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_46_19.png)
    


### Calculo da Matriz de Covariância


```python
# Calculo da matriz de covariância usando os retornos

cov_matrix = log_returns.cov()*252 # anualizando os dados
print(cov_matrix) # Fazer um gráfico bonitinho disso

```

              PETR4.SA  SBSP3.SA  RENT3.SA  ITUB4.SA  VALE3.SA
    PETR4.SA  0.112933  0.020214  0.021029  0.020938  0.022387
    SBSP3.SA  0.020214  0.086143  0.043596  0.022550  0.008943
    RENT3.SA  0.021029  0.043596  0.139612  0.037540  0.010981
    ITUB4.SA  0.020938  0.022550  0.037540  0.052941  0.012592
    VALE3.SA  0.022387  0.008943  0.010981  0.012592  0.090804
    


```python
# Cria um mapa de calor (heatmap) para visualizar a matriz de covariância
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Covariância')
plt.show()



```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_49_0.png)
    


# 6 - Modelagem e Previsão

## 6.1 Análise do Portfolio

### Encontrar o X-Day dos Retornos Históricos do Portfolio


```python
# Especificar um intervalo de confiança
confidence_interval = 0.95
test_windows = [30, 60, 90, 180, 252]  # Diferentes janelas de dias

```

### Calcular o VaR (Value at Risk)


```python
# Lista para armazenar os resultados do VaR
test_windows_results = []

# Loop para calcular o VaR para cada janela de tempo
for window in test_windows:
    range_returns = historical_returns.rolling(window=window).sum().dropna()

    if range_returns.empty:
        print(f'Janela de {window} dias sem dados suficientes.')
        continue

    VaR = -np.percentile(range_returns, 100 - (confidence_interval * 100)) * portfolio_value
    test_windows_results.append((window, VaR))
    print(f'\nVaR para janela de {window} dias: R$ {VaR:.2f}\n')

# Mostrar todos os resultados armazenados em test_windows_results
print("\nResultados do VaR para diferentes janelas de tempo:\n")
for window, var in test_windows_results:
    print(f"Janela de {window} dias: VaR = R$ {var:.2f}")

# Encontrar o melhor valor de VaR (o menor, pois ele representa a perda máxima)
best_window, best_VaR = min(test_windows_results, key=lambda x: x[1])
print(f'\nMelhor janela de tempo: {best_window} dias com VaR = R$ {best_VaR:.2f}\n')

# Plotar o gráfico da distribuição dos retornos do portfólio com a melhor janela
range_returns_best = historical_returns.rolling(window=best_window).sum().dropna()
plt.hist(range_returns_best * portfolio_value, bins=50, density=True, alpha=0.6, color='g')
plt.axvline(-best_VaR, color='r', linestyle='dashed', linewidth=2, label=f'VaR {confidence_interval*100}% de confiança')
plt.xlabel(f'{best_window} dias - Retorno do Portfolio (Reais)')
plt.ylabel('Frequência')
plt.title(f'Distribuição dos Retornos do Portfolio - {best_window} dias (Reais)')
plt.legend()
plt.show()
```

    
    VaR para janela de 30 dias: R$ 8564.20
    
    
    VaR para janela de 60 dias: R$ 9009.10
    
    
    VaR para janela de 90 dias: R$ 7769.46
    
    
    VaR para janela de 180 dias: R$ 846.81
    
    
    VaR para janela de 252 dias: R$ -540.24
    
    
    Resultados do VaR para diferentes janelas de tempo:
    
    Janela de 30 dias: VaR = R$ 8564.20
    Janela de 60 dias: VaR = R$ 9009.10
    Janela de 90 dias: VaR = R$ 7769.46
    Janela de 180 dias: VaR = R$ 846.81
    Janela de 252 dias: VaR = R$ -540.24
    
    Melhor janela de tempo: 252 dias com VaR = R$ -540.24
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_55_1.png)
    


### Backtesting


```python
# Escolha a melhor janela com base na análise anterior
# Por exemplo, se a janela de 180 dias for a escolhida

backtesting_days = best_window  # Período de backtesting (1 ano)
violacoes = []

for i in range(backtesting_days, len(historical_returns)):
    # Calcular o VaR com os dados anteriores ao dia i
    janela_treino = historical_returns[i-backtesting_days:i]
    range_returns = janela_treino.rolling(window=best_window).sum().dropna()

    if range_returns.empty:
        print(f'Janela de {best_window} dias sem dados suficientes no índice {i}.')
        continue  # Pula a iteração se range_returns estiver vazio

    VaR = -np.percentile(range_returns, 100 - (confidence_interval * 100)) * portfolio_value

    # Comparar com o retorno real
    retorno_real = historical_returns.iloc[i] * portfolio_value
    if retorno_real < -VaR:
        violacoes.append(1)
    else:
        violacoes.append(0)

# Imprimir a taxa de violação observada
taxa_violacao = np.mean(violacoes)
print(f'Taxa de Violação Observada: {taxa_violacao:.4f}')
```

    Taxa de Violação Observada: 0.9518
    

### Stress Test


```python
# Parâmetros do Stress Test
shock_factor = 0.5  # Aplicando um choque de 50% nos retornos (simula uma queda acentuada)

# Simular cenário de stress aplicando um choque negativo nos retornos
stress_scenario_returns = historical_returns.copy()
stress_scenario_returns *= np.random.uniform(1 - shock_factor, 1 - shock_factor, size=len(stress_scenario_returns))

# Calcular os retornos acumulados no cenário de stress
range_returns_stress = stress_scenario_returns.rolling(window=best_window).sum().dropna()

# Calcular o VaR no cenário de stress
VaR_stress = -np.percentile(range_returns_stress, 100 - (confidence_interval * 100)) * portfolio_value

# Exibir o resultado do VaR em cenário de stress
print(f'\nVaR em cenário de stress (choque de {shock_factor*100}%): R$ {VaR_stress:.2f}\n\n')

# Visualizar a distribuição dos retornos no cenário de stress
plt.hist(range_returns_stress * portfolio_value, bins=50, density=True, alpha=0.6, color='r')
plt.axvline(-VaR_stress, color='b', linestyle='dashed', linewidth=2, label=f'VaR Stress {confidence_interval*100}% de confiança')
plt.xlabel(f'{best_window} dias - Retorno do Portfolio (Reais) em Stress')
plt.ylabel('Frequência')
plt.title(f'Distribuição dos Retornos do Portfolio - {best_window} dias (Stress)')
plt.legend()
plt.show()
```

    
    VaR em cenário de stress (choque de 50.0%): R$ -270.12
    
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_59_1.png)
    


### Análise dos Retornos do Portfolio Escolhido


```python
# Calcula o retorno acumulado do portfolio
portfolio_return = (historical_returns.cumsum() + 1) * portfolio_value

# Calcula o retorno acumulado do CDI
cdi_return = (cdi['cdi'].cumsum() + 1) * portfolio_value

# Calcula o retorno acumulado do IBOV
ibov_return = (log_ibov['IBOV'].cumsum() + 1) * portfolio_value

# Obter os saldos atuais
saldo_atual_portfolio = portfolio_return[-1]
saldo_atual_cdi = cdi_return[-1]
saldo_atual_ibov = ibov_return[-1]

# Calcular as diferenças
diferenca_cdi = saldo_atual_portfolio - saldo_atual_cdi
diferenca_ibov = saldo_atual_portfolio - saldo_atual_ibov

# Imprimir os resultados
print('\n')
print(f"Saldo atual do Portfolio: R$ {saldo_atual_portfolio:.2f}")
print(f"Saldo atual do CDI: R$ {saldo_atual_cdi:.2f}")
print(f"Saldo atual do IBOV: R$ {saldo_atual_ibov:.2f}\n")
print(f"\nDiferença Portfolio vs. CDI: R$ {diferenca_cdi:.2f}")
print(f"Diferença Portfolio vs. IBOV: R$ {diferenca_ibov:.2f}\n")

# Plota os retornos acumulados
plt.figure(figsize=(12, 6))
plt.plot(historical_returns.index, portfolio_return, label='Portfolio')
plt.plot(cdi.index, cdi_return, label='CDI', linestyle='--')
plt.plot(log_ibov.index, ibov_return, label='IBOV', linestyle='-.')
plt.axhline(portfolio_value, color='k', linestyle=':', label='Investimento Inicial')
plt.title('Retorno Acumulado do Portfolio vs. Benchmarks (R$)')
plt.xlabel('Data')
plt.ylabel('Valor Acumulado (R$)')
plt.legend()
plt.grid(True)
plt.show()

```

    
    
    Saldo atual do Portfolio: R$ 140650.63
    Saldo atual do CDI: R$ 134261.80
    Saldo atual do IBOV: R$ 114376.66
    
    
    Diferença Portfolio vs. CDI: R$ 6388.83
    Diferença Portfolio vs. IBOV: R$ 26273.97
    
    

    C:\Users\jeand\AppData\Local\Temp\ipykernel_37080\2424190984.py:11: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      saldo_atual_portfolio = portfolio_return[-1]
    C:\Users\jeand\AppData\Local\Temp\ipykernel_37080\2424190984.py:12: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      saldo_atual_cdi = cdi_return[-1]
    C:\Users\jeand\AppData\Local\Temp\ipykernel_37080\2424190984.py:13: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      saldo_atual_ibov = ibov_return[-1]
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_61_2.png)
    



```python
# Calcula o retorno acumulado do portfolio
portfolio_return = (historical_returns.cumsum() + 1) * portfolio_value

# Calcula o retorno acumulado do CDI
cdi_return = (cdi['cdi'].cumsum() + 1) * portfolio_value

# Calcula o retorno acumulado do IBOV
ibov_return = (log_ibov['IBOV'].cumsum() + 1) * portfolio_value

# Plota os retornos acumulados
plt.figure(figsize=(12, 6))
plt.plot(historical_returns.index, portfolio_return, label='Portfolio')
plt.plot(cdi.index, cdi_return, label='CDI', linestyle='--')
plt.plot(log_ibov.index, ibov_return, label='IBOV', linestyle='-.')
plt.axhline(portfolio_value, color='k', linestyle=':', label='Investimento Inicial')
plt.title('Retorno Acumulado do Portfolio vs. Benchmarks (R$)')
plt.xlabel('Data')
plt.ylabel('Valor Acumulado (R$)')
plt.legend()
plt.grid(True)

# Calcula os valores finais
final_portfolio = portfolio_return.iloc[-1]
final_cdi = cdi_return.iloc[-1]
final_ibov = ibov_return.iloc[-1]

# Calcula as diferenças
diff_cdi = final_portfolio - final_cdi
diff_ibov = final_portfolio - final_ibov

# Adiciona os valores e diferenças no gráfico
plt.text(historical_returns.index[-1], final_portfolio, f'R$ {final_portfolio:,.2f}', ha='left', va='bottom')
plt.text(cdi.index[-1], final_cdi, f'R$ {final_cdi:,.2f}', ha='left', va='bottom')
plt.text(log_ibov.index[-1], final_ibov, f'R$ {final_ibov:,.2f}', ha='left', va='bottom')

plt.text(historical_returns.index[-1], final_portfolio/2, f'Dif. CDI: R$ {diff_cdi:,.2f}', ha='left', va='center', color='green' if diff_cdi > 0 else 'red')
plt.text(historical_returns.index[-1], final_portfolio/2 - 10000, f'Dif. IBOV: R$ {diff_ibov:,.2f}', ha='left', va='center', color='green' if diff_ibov > 0 else 'red')

plt.show()

```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_62_0.png)
    



```python
# Calcula as métricas para o portfolio
portfolio_mean_return = historical_returns.mean() * best_window  # Retorno médio anualizado
portfolio_std_dev = historical_returns.std() * np.sqrt(best_window)  # Desvio padrão anualizado
portfolio_sharpe_ratio = (portfolio_mean_return - risk_free_rate) / portfolio_std_dev  # Sharpe Ratio

# Calcula as métricas para o CDI
cdi_mean_return = cdi['cdi'].mean() * best_window
cdi_std_dev = cdi['cdi'].std() * np.sqrt(best_window)
cdi_sharpe_ratio = (cdi_mean_return - risk_free_rate) / cdi_std_dev  # Sharpe Ratio para o CDI (próximo de zero)

# Calcula as métricas para o IBOV
ibov_mean_return = log_ibov['IBOV'].mean() * best_window
ibov_std_dev = log_ibov['IBOV'].std() * np.sqrt(best_window)
ibov_sharpe_ratio = (ibov_mean_return - risk_free_rate) / ibov_std_dev

# Cria um DataFrame para exibir as métricas
metrics_df = pd.DataFrame({
    'Métrica': ['Retorno Médio Anualizado', 'Desvio Padrão Anualizado', 'Sharpe Ratio'],
    'Portfolio': [portfolio_mean_return, portfolio_std_dev, portfolio_sharpe_ratio],
    'CDI': [cdi_mean_return, cdi_std_dev, cdi_sharpe_ratio],
    'IBOV': [ibov_mean_return, ibov_std_dev, ibov_sharpe_ratio]
})

# Exibe o DataFrame
print(metrics_df.to_string(index=False))

```

                     Métrica  Portfolio        CDI     IBOV
    Retorno Médio Anualizado   0.136586   0.114509 0.048370
    Desvio Padrão Anualizado   0.192119   0.000788 0.173979
                Sharpe Ratio   0.708577 144.766572 0.275406
    

## 6.2 Previsão do Portfolio

### Portfolio - CAPM


```python
# Apaga primeira linha de historical_returns

historical_returns = historical_returns.iloc[1:]
historical_returns

```




    Date
    2021-12-29   -0.009281
    2021-12-30    0.002959
    2022-01-03    0.000723
    2022-01-04   -0.000473
    2022-01-05   -0.024498
                    ...   
    2024-12-18   -0.027317
    2024-12-19    0.011619
    2024-12-20    0.009274
    2024-12-23   -0.003756
    2024-12-26    0.001431
    Name: returns, Length: 749, dtype: float64




```python
# Calcula o retorno médio do portfolio
portfolio_mean_return = historical_returns.mean() * best_window

# Calcula o retorno médio do IBOV
ibov_mean_return = log_ibov['IBOV'].mean() * best_window

# Calcula a covariância entre o portfolio e o IBOV
cov_portfolio_ibov = historical_returns.cov(log_ibov['IBOV']) * best_window

# Calcula a variância do IBOV
var_ibov = log_ibov['IBOV'].var() * best_window

# Calcula o beta do portfolio
beta_portfolio = cov_portfolio_ibov / var_ibov

# Calcula o retorno esperado do portfolio usando o CAPM
expected_return_capm = risk_free_rate + beta_portfolio * (ibov_mean_return - risk_free_rate)

# Calcula o alfa do portfolio
alpha_portfolio = portfolio_mean_return - expected_return_capm

# Imprime os resultados
print(f"Beta do Portfolio: {beta_portfolio:.4f}")
print(f"Retorno Esperado (CAPM): {expected_return_capm:.4f}")
print(f"Alfa do Portfolio: {alpha_portfolio:.4f}")

```

    Beta do Portfolio: 1.0140
    Retorno Esperado (CAPM): 0.0490
    Alfa do Portfolio: 0.0877
    


```python

# Calcula o erro de previsão do CAPM
capm_prediction_error = portfolio_mean_return - expected_return_capm

# Calcula o R-quadrado do modelo CAPM (quanto da variância do retorno do portfolio é explicada pelo IBOV)
r_squared = (cov_portfolio_ibov ** 2) / (var_ibov * historical_returns.var() * best_window)

# Imprime os resultados adicionais
print(f"Erro de Previsão do CAPM: {capm_prediction_error:.4f}")
print(f"R-quadrado do Modelo CAPM: {r_squared:.4f}")

# Interpretação dos resultados
print("\nInterpretação dos Resultados:")
print(f"- Beta do Portfolio ({beta_portfolio:.4f}): Indica que o portfolio é {beta_portfolio:.2f} vezes mais volátil que o IBOV.")
if alpha_portfolio > 0:
    print(f"- Alfa do Portfolio ({alpha_portfolio:.4f}): Positivo, indicando que o portfolio gerou retornos acima do esperado pelo CAPM, sugerindo uma possível habilidade do gestor.")
elif alpha_portfolio < 0:
    print(f"- Alfa do Portfolio ({alpha_portfolio:.4f}): Negativo, indicando que o portfolio gerou retornos abaixo do esperado pelo CAPM.")
else:
    print(f"- Alfa do Portfolio ({alpha_portfolio:.4f}): Zero, indicando que o portfolio gerou retornos em linha com o esperado pelo CAPM.")
print(f"- R-quadrado do Modelo CAPM ({r_squared:.4f}): Indica que {r_squared*100:.2f}% da variância do retorno do portfolio é explicada pelo IBOV.")

# Visualização dos resultados
plt.figure(figsize=(10, 6))
plt.scatter(log_ibov['IBOV'], historical_returns, alpha=0.6)
plt.xlabel('Retorno do IBOV')
plt.ylabel('Retorno do Portfolio')
plt.title('Relação entre Retorno do Portfolio e Retorno do IBOV')

# Adicionar a linha de regressão (CAPM)
x = np.linspace(log_ibov['IBOV'].min(), log_ibov['IBOV'].max(), 100)
y = risk_free_rate + beta_portfolio * (x - risk_free_rate)
plt.plot(x, y, color='red', label='Linha de Regressão (CAPM)')

plt.legend()
plt.grid(True)
plt.show()

```

    Erro de Previsão do CAPM: 0.0877
    R-quadrado do Modelo CAPM: 0.8421
    
    Interpretação dos Resultados:
    - Beta do Portfolio (1.0140): Indica que o portfolio é 1.01 vezes mais volátil que o IBOV.
    - Alfa do Portfolio (0.0877): Positivo, indicando que o portfolio gerou retornos acima do esperado pelo CAPM, sugerindo uma possível habilidade do gestor.
    - R-quadrado do Modelo CAPM (0.8421): Indica que 84.21% da variância do retorno do portfolio é explicada pelo IBOV.
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_68_1.png)
    


### Portfolio - Regressão Linear


```python
df_modelagem = log_returns.copy()
```


```python
# Preparação inicial
valor_investido = portfolio_value  # Exemplo: R$100.000
dias_previsao = best_window  # Quantidade de dias úteis para prever no futuro

# Junta a df_modelagem, o cdi, e ibov, usando o index de date de df_modelagem como base
df_modelagem = df_modelagem.join(cdi, how='left').join(log_ibov, how='left')
df_modelagem = df_modelagem.join(historical_returns, how = 'left')
df_modelagem.dropna(inplace=True)

# Criando uma coluna ordinal para as datas no DataFrame
df_modelagem['Date_ordinal'] = df_modelagem.index.map(lambda x: x.toordinal())

# Executando regressão linear para retornos, IBOV e CDI
resultados = {}
for coluna in ['returns', 'IBOV', 'cdi']:
    resultados[coluna] = regressao_linear(df_modelagem, coluna , best_window=best_window)

# Calculando saldo acumulado histórico e futuro para cada série
saldos = {'historico': {}, 'futuro': {}}
for coluna in ['returns', 'IBOV', 'cdi']:
    # Saldo histórico
    saldo_historico = [valor_investido]
    for retorno in df_modelagem[coluna]:
        saldo_historico.append(saldo_historico[-1] * (1 + retorno))
    saldos['historico'][coluna] = saldo_historico[1:]
    
    # Saldo futuro
    saldo_futuro = [saldos['historico'][coluna][-1]]
    for retorno in resultados[coluna]['previsoes_futuras']:
        saldo_futuro.append(saldo_futuro[-1] * (1 + retorno))
    saldos['futuro'][coluna] = saldo_futuro[1:]

# Plotando os resultados
plt.figure(figsize=(14, 8))

# Histórico
for coluna, cor in zip(['returns', 'IBOV', 'cdi'], ['blue', 'green', 'orange']):
    plt.plot(df_modelagem.index, saldos['historico'][coluna], label=f'Saldo Histórico ({coluna.upper()})', color=cor)

# Futuro
for coluna, cor in zip(['returns', 'IBOV', 'cdi'], ['blue', 'green', 'orange']):
    plt.plot(resultados[coluna]['datas_futuras'], saldos['futuro'][coluna], linestyle='--', label=f'Saldo Previsto ({coluna.upper()})', color=cor)

# Linha do investimento inicial
plt.axhline(y=valor_investido, color='red', linestyle='--', label='Saldo Inicial (Investimento)')

# Configurações do gráfico
plt.xlabel('Data')
plt.ylabel('Saldo (R$)')
plt.title('Comparação de Saldo: Retornos, IBOV e CDI (Histórico e Previsão)')
plt.legend()
plt.grid(True)
plt.show()

# Métricas da regressão
for coluna in ['returns', 'IBOV', 'cdi']:
    print(f"\nMétricas para {coluna.upper()}:")
    print(f"  - MSE: {resultados[coluna]['mse']:.6f}")
    print(f"  - MAE: {resultados[coluna]['mae']:.6f}")
    print(f"  - R²: {resultados[coluna]['r2']:.6f}")

```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_71_0.png)
    


    
    Métricas para RETURNS:
      - MSE: 0.000146
      - MAE: 0.009270
      - R²: 0.001720
    
    Métricas para IBOV:
      - MSE: 0.000120
      - MAE: 0.008454
      - R²: 0.000258
    
    Métricas para CDI:
      - MSE: 0.000000
      - MAE: 0.000038
      - R²: 0.147756
    

### Previsão do Portfolio - ARIMA

#### Portfólio


```python
# análise dos retornos

# Calcula o retorno acumulado do portfolio
portfolio_return = (historical_returns.cumsum() + 1) * portfolio_value
print(type(portfolio_return))
print(portfolio_return)
portfolio_return.plot()

# fazendo a decomposição
decomposition = seasonal_decompose(portfolio_return.dropna(), model='additive', period=best_window)

# Plotar a decomposição
fig, axes = plt.subplots(4, 1, figsize=(12, 12))
decomposition.observed.plot(ax=axes[0], title='Observado')
decomposition.trend.plot(ax=axes[1], title='Tendência')
decomposition.seasonal.plot(ax=axes[2], title='Sazonalidade')
decomposition.resid.plot(ax=axes[3], title='Resíduos')
plt.tight_layout()
plt.show()

# Teste de normalidade dos resíduos (Shapiro-Wilk)
statistic, p_value = shapiro(decomposition.resid.dropna())
print(f'\nTeste de Shapiro-Wilk para retornos do portfolio: Estatística={statistic:.3f}, p-valor={p_value:.3f}')
if p_value > 0.05:
    print('Provavelmente normal')
else:
    print('Provavelmente não normal')

# Teste de estacionariedade (Dickey-Fuller Aumentado)
result_adfuller = adfuller(portfolio_return.dropna())
print(f'\nTeste de Dickey-Fuller Aumentado para retornos do portfolio:')
print('ADF Statistic: %f' % result_adfuller[0])
print('p-value: %f' % result_adfuller[1])
print('Critical Values:')
for key, value in result_adfuller[4].items():
    print('\t%s: %.3f' % (key, value))

# Verifique se o p-valor do teste ADF é menor que 0.05
if result_adfuller[1] <= 0.05:
    print('Série provavelmente estacionária\n')
else:
    print('Série provavelmente não estacionária\n')



# Autocorrelação e Autocorrelação Parcial dos resíduos
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(decomposition.resid.dropna(), ax=axes[0])
plot_pacf(decomposition.resid.dropna(), ax=axes[1])
plt.tight_layout()
plt.show()
```

    <class 'pandas.core.series.Series'>
    Date
    2021-12-29     99071.921171
    2021-12-30     99367.847829
    2022-01-03     99440.143549
    2022-01-04     99392.851153
    2022-01-05     96943.004342
                      ...      
    2024-12-18    138793.858439
    2024-12-19    139955.749417
    2024-12-20    140883.137018
    2024-12-23    140507.538430
    2024-12-26    140650.629150
    Name: returns, Length: 749, dtype: float64
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_74_1.png)
    



    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_74_2.png)
    


    
    Teste de Shapiro-Wilk para retornos do portfolio: Estatística=0.985, p-valor=0.000
    Provavelmente não normal
    
    Teste de Dickey-Fuller Aumentado para retornos do portfolio:
    ADF Statistic: -2.210533
    p-value: 0.202445
    Critical Values:
    	1%: -3.439
    	5%: -2.865
    	10%: -2.569
    Série provavelmente não estacionária
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_74_4.png)
    



```python

# Ajustar o modelo Auto-ARIMA
modelo_auto_arima = auto_arima(portfolio_return,
                               start_p=1, start_q=1,
                               max_p=15, max_q=15,
                               d=0,  # Determina d automaticamente
                               seasonal=True,  # Ajuste conforme necessidade
                               trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)

print(modelo_auto_arima.summary())

# Dividir os dados em treino e teste (80% treino, 20% teste)
train_size = int(len(portfolio_return) * 0.8)
train_data, test_data = portfolio_return[:train_size], portfolio_return[train_size:]

# Imprimir os shapes dos conjuntos de treino e teste
print("Shape dos dados de treino:", train_data.shape)
print("Shape dos dados de teste:", test_data.shape)

model_ARIMA = ARIMA(train_data, order=(3,0,2))
result = model_ARIMA.fit()
print(result.summary())

# Realizar previsões no conjunto de teste
predictions_test = result.get_forecast(steps=len(test_data))
pred_conf_int = predictions_test.conf_int()
predicted_mean = predictions_test.predicted_mean

# Prever o melhor x-day dias no futuro
forecast_steps = best_window
portfolio_forecast = result.get_forecast(steps=forecast_steps)
forecast_conf_int = portfolio_forecast.conf_int()
forecast_mean = portfolio_forecast.predicted_mean

# Plotar gráficos

# Gráfico de comparação do teste
plt.figure(figsize=(14, 7))
#plt.plot(portfolio_return.index[:len(train_data)].index, train_data, label ="Treino", color = 'green')
plt.plot(portfolio_return.index[-len(test_data):], test_data, label='Real', color='blue')
plt.plot(portfolio_return.index[-len(test_data):], predicted_mean, label='Previsto (Teste)', color='red')
plt.fill_between(portfolio_return.index[-len(test_data):], pred_conf_int.iloc[:, 0], pred_conf_int.iloc[:, 1], color='orange', alpha=0.3, label='Intervalo de Confiança (Teste)')
plt.xlabel('Data')
plt.ylabel('Saldo')
plt.title('Comparação entre Real e Previsto no Teste')
plt.legend()
plt.grid(True)
plt.show()


# Criar um índice de datas para a previsão
last_date = portfolio_return.index[-1]
forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='B')  # 'B' para dias úteis

# Plotar o gráfico da previsão
plt.figure(figsize=(14, 7))
plt.plot(portfolio_return.index, portfolio_return, label='Histórico', color='blue')
plt.plot(forecast_index, forecast_mean, label='Previsão', color='red')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Intervalo de Confiança')
plt.xlabel('Data')
plt.ylabel('Saldo')
plt.title('Previsão do Retorno do Portfolio')
plt.legend()
plt.grid(True)
plt.show()
```

    Performing stepwise search to minimize aic
     ARIMA(1,0,1)(0,0,0)[0] intercept   : AIC=12773.777, Time=0.07 sec
     ARIMA(0,0,0)(0,0,0)[0] intercept   : AIC=16521.738, Time=0.01 sec
     ARIMA(1,0,0)(0,0,0)[0] intercept   : AIC=inf, Time=0.07 sec
     ARIMA(0,0,1)(0,0,0)[0] intercept   : AIC=15568.263, Time=0.24 sec
     ARIMA(0,0,0)(0,0,0)[0]             : AIC=19789.274, Time=0.02 sec
     ARIMA(2,0,1)(0,0,0)[0] intercept   : AIC=12773.053, Time=0.40 sec
     ARIMA(2,0,0)(0,0,0)[0] intercept   : AIC=inf, Time=0.09 sec
     ARIMA(3,0,1)(0,0,0)[0] intercept   : AIC=12774.687, Time=0.13 sec
     ARIMA(2,0,2)(0,0,0)[0] intercept   : AIC=12775.358, Time=0.23 sec
     ARIMA(1,0,2)(0,0,0)[0] intercept   : AIC=12775.038, Time=0.11 sec
     ARIMA(3,0,0)(0,0,0)[0] intercept   : AIC=inf, Time=0.12 sec
     ARIMA(3,0,2)(0,0,0)[0] intercept   : AIC=12768.565, Time=0.65 sec
     ARIMA(4,0,2)(0,0,0)[0] intercept   : AIC=12770.315, Time=0.83 sec
     ARIMA(3,0,3)(0,0,0)[0] intercept   : AIC=12770.323, Time=0.59 sec
     ARIMA(2,0,3)(0,0,0)[0] intercept   : AIC=12774.832, Time=0.40 sec
     ARIMA(4,0,1)(0,0,0)[0] intercept   : AIC=12775.235, Time=0.53 sec
     ARIMA(4,0,3)(0,0,0)[0] intercept   : AIC=12772.251, Time=0.66 sec
     ARIMA(3,0,2)(0,0,0)[0]             : AIC=inf, Time=0.55 sec
    
    Best model:  ARIMA(3,0,2)(0,0,0)[0] intercept
    Total fit time: 5.718 seconds
                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                  749
    Model:               SARIMAX(3, 0, 2)   Log Likelihood               -6377.283
    Date:                Thu, 26 Dec 2024   AIC                          12768.565
    Time:                        23:15:03   BIC                          12800.897
    Sample:                             0   HQIC                         12781.024
                                    - 749                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    intercept    935.6730    994.243      0.941      0.347   -1013.008    2884.354
    ar.L1         -0.0286      0.125     -0.228      0.820      -0.275       0.217
    ar.L2          0.2558      0.126      2.036      0.042       0.010       0.502
    ar.L3          0.7653      0.116      6.574      0.000       0.537       0.994
    ma.L1          1.1027      0.125      8.806      0.000       0.857       1.348
    ma.L2          0.7939      0.111      7.175      0.000       0.577       1.011
    sigma2      1.441e+06      0.563   2.56e+06      0.000    1.44e+06    1.44e+06
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.21   Jarque-Bera (JB):               116.20
    Prob(Q):                              0.64   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.41   Skew:                             0.13
    Prob(H) (two-sided):                  0.00   Kurtosis:                         4.91
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    [2] Covariance matrix is singular or near-singular, with condition number 1.44e+23. Standard errors may be unstable.
    Shape dos dados de treino: (599,)
    Shape dos dados de teste: (150,)
    

    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\statespace\sarimax.py:966: UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.
      warn('Non-stationary starting autoregressive parameters'
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\statespace\sarimax.py:978: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.
      warn('Non-invertible starting MA parameters found.'
    

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                returns   No. Observations:                  599
    Model:                 ARIMA(3, 0, 2)   Log Likelihood               -5121.476
    Date:                Thu, 26 Dec 2024   AIC                          10256.951
    Time:                        23:15:04   BIC                          10287.718
    Sample:                             0   HQIC                         10268.929
                                    - 599                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       1.271e+05    1.6e+04      7.920      0.000    9.56e+04    1.59e+05
    ar.L1          0.0596      0.146      0.409      0.682      -0.226       0.345
    ar.L2          0.1718      0.128      1.344      0.179      -0.079       0.422
    ar.L3          0.7618      0.122      6.219      0.000       0.522       1.002
    ma.L1          1.0207      0.147      6.964      0.000       0.733       1.308
    ma.L2          0.7721      0.125      6.182      0.000       0.527       1.017
    sigma2      1.549e+06   7.13e+04     21.731      0.000    1.41e+06    1.69e+06
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):                88.80
    Prob(Q):                              0.92   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.43   Skew:                             0.22
    Prob(H) (two-sided):                  0.00   Kurtosis:                         4.84
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    

    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_75_4.png)
    



    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_75_5.png)
    


#### CDI


```python

# Supondo que 'historical_CDI' seja uma série temporal de retornos do CDI
cdi_return = (log_cdi.cumsum() + 1) * portfolio_value
print(type(cdi_return))
print(cdi_return)
cdi_return.plot()

# Decomposição sazonal para CDI
decomposition_cdi = seasonal_decompose(cdi_return.dropna(), model='additive', period=252)

# Plotar a decomposição
fig, axes = plt.subplots(4, 1, figsize=(12, 12))
decomposition_cdi.observed.plot(ax=axes[0], title='Observado')
decomposition_cdi.trend.plot(ax=axes[1], title='Tendência')
decomposition_cdi.seasonal.plot(ax=axes[2], title='Sazonalidade')
decomposition_cdi.resid.plot(ax=axes[3], title='Resíduos')
plt.tight_layout()
plt.show()

# Teste de normalidade dos resíduos (Shapiro-Wilk)
statistic, p_value = shapiro(decomposition_cdi.resid.dropna())
print(f'\nTeste de Shapiro-Wilk para CDI: Estatística={statistic:.3f}, p-valor={p_value:.3f}')
if p_value > 0.05:
    print('Provavelmente normal')
else:
    print('Provavelmente não normal')

# Teste de estacionariedade (Dickey-Fuller Aumentado)
result_adfuller_cdi = adfuller(cdi_return.dropna())
print(f'\nTeste de Dickey-Fuller Aumentado para CDI:')
print('ADF Statistic: %f' % result_adfuller_cdi[0])
print('p-value: %f' % result_adfuller_cdi[1])
print('Critical Values:')
for key, value in result_adfuller_cdi[4].items():
    print('\t%s: %.3f' % (key, value))
if result_adfuller_cdi[1] <= 0.05:
    print('Série provavelmente estacionária\n')
else:
    print('Série provavelmente não estacionária\n')

# Autocorrelação e Autocorrelação Parcial dos resíduos
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(decomposition_cdi.resid.dropna(), ax=axes[0])
plot_pacf(decomposition_cdi.resid.dropna(), ax=axes[1])
plt.tight_layout()
plt.show()

# Ajustar o modelo Auto-ARIMA para CDI
modelo_auto_arima_cdi = auto_arima(cdi_return,
                                   start_p=1, start_q=1,
                                   max_p=15, max_q=15,
                                   d=0,  # Determina d automaticamente
                                   seasonal=True,  # Ajuste conforme necessidade
                                   trace=True,
                                   error_action='ignore',
                                   suppress_warnings=True,
                                   stepwise=True)
print(modelo_auto_arima_cdi.summary())

# Dividir os dados em treino e teste (80% treino, 20% teste) para CDI
train_size_cdi = int(len(cdi_return) * 0.8)
train_data_cdi, test_data_cdi = cdi_return[:train_size_cdi], cdi_return[train_size_cdi:]

# Imprimir os shapes dos conjuntos de treino e teste
print("Shape dos dados de treino CDI:", train_data_cdi.shape)
print("Shape dos dados de teste CDI:", test_data_cdi.shape)

# Ajuste o modelo ARIMA para CDI
model_ARIMA_cdi = ARIMA(train_data_cdi, order=(2,0,1))
result_cdi = model_ARIMA_cdi.fit()
result_cdi.summary()

# Realizar previsões no conjunto de teste
predictions_test_cdi = result_cdi.get_forecast(steps=len(test_data_cdi))
pred_conf_int_cdi = predictions_test_cdi.conf_int()
predicted_mean_cdi = predictions_test_cdi.predicted_mean

# Prever o melhor x-day dias no futuro
forecast_steps_cdi = best_window
forecast_cdi = result_cdi.get_forecast(steps=forecast_steps_cdi)
forecast_conf_int_cdi = forecast_cdi.conf_int()
forecast_mean_cdi = forecast_cdi.predicted_mean

# Plotar gráfico de comparação do teste para CDI
plt.figure(figsize=(14, 7))
plt.plot(cdi_return.index[-len(test_data_cdi):], test_data_cdi, label='Real', color='blue')
plt.plot(cdi_return.index[-len(test_data_cdi):], predicted_mean_cdi, label='Previsto (Teste)', color='red')
plt.fill_between(cdi_return.index[-len(test_data_cdi):], pred_conf_int_cdi.iloc[:, 0], pred_conf_int_cdi.iloc[:, 1], color='orange', alpha=0.3, label='Intervalo de Confiança (Teste)')
plt.xlabel('Data')
plt.ylabel('Saldo')
plt.title('Comparação entre Real e Previsto no Teste - CDI')
plt.legend()
plt.grid(True)
plt.show()

# Criar um índice de datas para a previsão
last_date = cdi_return.index[-1]
forecast_index_cdi = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps_cdi, freq='B')  # 'B' para dias úteis

# Plotar o gráfico da previsão
plt.figure(figsize=(14, 7))
plt.plot(cdi_return.index, cdi_return, label='Histórico', color='blue')
plt.plot(forecast_index_cdi, forecast_mean_cdi, label='Previsão', color='red')
plt.fill_between(forecast_index_cdi, forecast_conf_int_cdi.iloc[:, 0], forecast_conf_int_cdi.iloc[:, 1], color='pink', alpha=0.3, label='Intervalo de Confiança')
plt.xlabel('Data')
plt.ylabel('Saldo')
plt.title('Previsão do Retorno do CDI')
plt.legend()
plt.grid(True)
plt.show()

```

    <class 'pandas.core.frame.DataFrame'>
                          cdi
    data                     
    2021-12-28  100000.000000
    2021-12-29  100000.000000
    2021-12-30  100000.000000
    2021-12-31  100000.000000
    2022-01-03  100000.000000
    ...                   ...
    2024-12-18  126984.720482
    2024-12-19  126984.720482
    2024-12-20  126984.720482
    2024-12-23  126984.720482
    2024-12-24  126984.720482
    
    [753 rows x 1 columns]
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_77_1.png)
    



    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_77_2.png)
    


    
    Teste de Shapiro-Wilk para CDI: Estatística=0.982, p-valor=0.000
    Provavelmente não normal
    
    Teste de Dickey-Fuller Aumentado para CDI:
    ADF Statistic: -2.365285
    p-value: 0.151775
    Critical Values:
    	1%: -3.439
    	5%: -2.865
    	10%: -2.569
    Série provavelmente não estacionária
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_77_4.png)
    


    Performing stepwise search to minimize aic
     ARIMA(1,0,1)(0,0,0)[0] intercept   : AIC=12331.247, Time=0.07 sec
     ARIMA(0,0,0)(0,0,0)[0] intercept   : AIC=16179.839, Time=0.01 sec
     ARIMA(1,0,0)(0,0,0)[0] intercept   : AIC=inf, Time=0.09 sec
     ARIMA(0,0,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.15 sec
     ARIMA(0,0,0)(0,0,0)[0]             : AIC=19834.212, Time=0.01 sec
     ARIMA(2,0,1)(0,0,0)[0] intercept   : AIC=12328.707, Time=0.52 sec
     ARIMA(2,0,0)(0,0,0)[0] intercept   : AIC=inf, Time=0.09 sec
     ARIMA(3,0,1)(0,0,0)[0] intercept   : AIC=12330.842, Time=0.68 sec
     ARIMA(2,0,2)(0,0,0)[0] intercept   : AIC=12330.951, Time=0.28 sec
     ARIMA(1,0,2)(0,0,0)[0] intercept   : AIC=12333.389, Time=0.10 sec
     ARIMA(3,0,0)(0,0,0)[0] intercept   : AIC=inf, Time=0.14 sec
     ARIMA(3,0,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.78 sec
     ARIMA(2,0,1)(0,0,0)[0]             : AIC=inf, Time=0.23 sec
    
    Best model:  ARIMA(2,0,1)(0,0,0)[0] intercept
    Total fit time: 3.156 seconds
                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                  753
    Model:               SARIMAX(2, 0, 1)   Log Likelihood               -6159.353
    Date:                Thu, 26 Dec 2024   AIC                          12328.707
    Time:                        23:15:08   BIC                          12351.827
    Sample:                             0   HQIC                         12337.614
                                    - 753                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    intercept    886.2907    646.650      1.371      0.171    -381.120    2153.701
    ar.L1          0.0228      0.274      0.083      0.934      -0.515       0.561
    ar.L2          0.9699      0.272      3.564      0.000       0.437       1.503
    ma.L1          0.9762      0.251      3.894      0.000       0.485       1.468
    sigma2      7.369e+05     57.143   1.29e+04      0.000    7.37e+05    7.37e+05
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.03   Jarque-Bera (JB):            606016.50
    Prob(Q):                              0.86   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.35   Skew:                             9.10
    Prob(H) (two-sided):                  0.00   Kurtosis:                       140.78
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    [2] Covariance matrix is singular or near-singular, with condition number 1.17e+18. Standard errors may be unstable.
    Shape dos dados de treino CDI: (602, 1)
    Shape dos dados de teste CDI: (151, 1)
    

    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\statespace\sarimax.py:966: UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.
      warn('Non-stationary starting autoregressive parameters'
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\statespace\sarimax.py:978: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.
      warn('Non-invertible starting MA parameters found.'
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_77_7.png)
    



    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_77_8.png)
    


#### IBOV


```python
# Análise do IBOV


# Supondo que 'historical_IBOV' seja uma série temporal de retornos do IBOV
ibov_return = (log_ibov.cumsum() + 1) * portfolio_value
print(type(ibov_return))
print(ibov_return)
ibov_return.plot()

# Decomposição sazonal para IBOV
decomposition_ibov = seasonal_decompose(ibov_return.dropna(), model='additive', period=252)

# Plotar a decomposição
fig, axes = plt.subplots(4, 1, figsize=(12, 12))
decomposition_ibov.observed.plot(ax=axes[0], title='Observado')
decomposition_ibov.trend.plot(ax=axes[1], title='Tendência')
decomposition_ibov.seasonal.plot(ax=axes[2], title='Sazonalidade')
decomposition_ibov.resid.plot(ax=axes[3], title='Resíduos')
plt.tight_layout()
plt.show()

# Teste de normalidade dos resíduos (Shapiro-Wilk)
statistic, p_value = shapiro(decomposition_ibov.resid.dropna())
print(f'\nTeste de Shapiro-Wilk para IBOV: Estatística={statistic:.3f}, p-valor={p_value:.3f}')
if p_value > 0.05:
    print('Provavelmente normal')
else:
    print('Provavelmente não normal')

# Teste de estacionariedade (Dickey-Fuller Aumentado)
result_adfuller_ibov = adfuller(ibov_return.dropna())
print(f'\nTeste de Dickey-Fuller Aumentado para IBOV:')
print('ADF Statistic: %f' % result_adfuller_ibov[0])
print('p-value: %f' % result_adfuller_ibov[1])
print('Critical Values:')
for key, value in result_adfuller_ibov[4].items():
    print('\t%s: %.3f' % (key, value))
if result_adfuller_ibov[1] <= 0.05:
    print('Série provavelmente estacionária\n')
else:
    print('Série provavelmente não estacionária\n')

# Autocorrelação e Autocorrelação Parcial dos resíduos
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(decomposition_ibov.resid.dropna(), ax=axes[0])
plot_pacf(decomposition_ibov.resid.dropna(), ax=axes[1])
plt.tight_layout()
plt.show()

# Ajustar o modelo Auto-ARIMA para IBOV
modelo_auto_arima_ibov = auto_arima(ibov_return,
                                    start_p=1, start_q=1,
                                    max_p=15, max_q=15,
                                    d=0,  # Determina d automaticamente
                                    seasonal=True,  # Ajuste conforme necessidade
                                    trace=True,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True)
print(modelo_auto_arima_ibov.summary())

# Dividir os dados em treino e teste (80% treino, 20% teste) para IBOV
train_size_ibov = int(len(ibov_return) * 0.8)
train_data_ibov, test_data_ibov = ibov_return[:train_size_ibov], ibov_return[train_size_ibov:]

# Imprimir os shapes dos conjuntos de treino e teste
print("Shape dos dados de treino IBOV:", train_data_ibov.shape)
print("Shape dos dados de teste IBOV:", test_data_ibov.shape)

# Ajuste o modelo ARIMA para IBOV
model_ARIMA_ibov = ARIMA(train_data_ibov, order=(1,0,2))
result_ibov = model_ARIMA_ibov.fit()
result_ibov.summary()

# Realizar previsões no conjunto de teste
predictions_test_ibov = result_ibov.get_forecast(steps=len(test_data_ibov))
pred_conf_int_ibov = predictions_test_ibov.conf_int()
predicted_mean_ibov = predictions_test_ibov.predicted_mean

# Prever o melhor x-day dias no futuro
forecast_steps_ibov = best_window
forecast_ibov = result_ibov.get_forecast(steps=forecast_steps_ibov)
forecast_conf_int_ibov = forecast_ibov.conf_int()
forecast_mean_ibov = forecast_ibov.predicted_mean

# Plotar gráfico de comparação do teste para IBOV
plt.figure(figsize=(14, 7))
plt.plot(ibov_return.index[-len(test_data_ibov):], test_data_ibov, label='Real', color='blue')
plt.plot(ibov_return.index[-len(test_data_ibov):], predicted_mean_ibov, label='Previsto (Teste)', color='red')
plt.fill_between(ibov_return.index[-len(test_data_ibov):], pred_conf_int_ibov.iloc[:, 0], pred_conf_int_ibov.iloc[:, 1], color='orange', alpha=0.3, label='Intervalo de Confiança (Teste)')
plt.xlabel('Data')
plt.ylabel('Saldo')
plt.title('Comparação entre Real e Previsto no Teste - IBOV')
plt.legend()
plt.grid(True)
plt.show()


# Criar um índice de datas para a previsão
last_date = ibov_return.index[-1]
forecast_index_ibov = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps_ibov, freq='B')  # 'B' para dias úteis

# Plotar o gráfico da previsão
plt.figure(figsize=(14, 7))
plt.plot(ibov_return.index, ibov_return, label='Histórico', color='blue')
plt.plot(forecast_index_ibov, forecast_mean_ibov, label='Previsão', color='red')
plt.fill_between(forecast_index_ibov, forecast_conf_int_ibov.iloc[:, 0], forecast_conf_int_ibov.iloc[:, 1], color='pink', alpha=0.3, label='Intervalo de Confiança')
plt.xlabel('Data')
plt.ylabel('Saldo')
plt.title('Previsão do Retorno do CDI')
plt.legend()
plt.grid(True)
plt.show()

```

    <class 'pandas.core.frame.DataFrame'>
                         IBOV
    Date                     
    2021-12-29   99275.494388
    2021-12-30   99959.940100
    2022-01-03   99097.634523
    2022-01-04   98704.259674
    2022-01-05   96251.564850
    ...                   ...
    2024-12-18  114124.019804
    2024-12-19  114467.878632
    2024-12-20  115219.248848
    2024-12-23  114119.879686
    2024-12-26  114376.656392
    
    [749 rows x 1 columns]
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_79_1.png)
    



    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_79_2.png)
    


    
    Teste de Shapiro-Wilk para IBOV: Estatística=0.980, p-valor=0.000
    Provavelmente não normal
    
    Teste de Dickey-Fuller Aumentado para IBOV:
    ADF Statistic: -1.993790
    p-value: 0.289303
    Critical Values:
    	1%: -3.439
    	5%: -2.865
    	10%: -2.569
    Série provavelmente não estacionária
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_79_4.png)
    


    Performing stepwise search to minimize aic
     ARIMA(1,0,1)(0,0,0)[0] intercept   : AIC=12616.728, Time=0.06 sec
     ARIMA(0,0,0)(0,0,0)[0] intercept   : AIC=15692.877, Time=0.01 sec
     ARIMA(1,0,0)(0,0,0)[0] intercept   : AIC=inf, Time=0.03 sec
     ARIMA(0,0,1)(0,0,0)[0] intercept   : AIC=14784.122, Time=0.17 sec
     ARIMA(0,0,0)(0,0,0)[0]             : AIC=19532.964, Time=0.01 sec
     ARIMA(2,0,1)(0,0,0)[0] intercept   : AIC=12617.348, Time=0.37 sec
     ARIMA(1,0,2)(0,0,0)[0] intercept   : AIC=12616.652, Time=0.10 sec
     ARIMA(0,0,2)(0,0,0)[0] intercept   : AIC=14130.501, Time=0.29 sec
     ARIMA(2,0,2)(0,0,0)[0] intercept   : AIC=12618.397, Time=0.25 sec
     ARIMA(1,0,3)(0,0,0)[0] intercept   : AIC=12618.267, Time=0.12 sec
     ARIMA(0,0,3)(0,0,0)[0] intercept   : AIC=13769.348, Time=0.76 sec
     ARIMA(2,0,3)(0,0,0)[0] intercept   : AIC=12620.263, Time=0.35 sec
     ARIMA(1,0,2)(0,0,0)[0]             : AIC=12622.635, Time=0.24 sec
    
    Best model:  ARIMA(1,0,2)(0,0,0)[0] intercept
    Total fit time: 2.770 seconds
                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                  749
    Model:               SARIMAX(1, 0, 2)   Log Likelihood               -6303.326
    Date:                Thu, 26 Dec 2024   AIC                          12616.652
    Time:                        23:15:12   BIC                          12639.746
    Sample:                             0   HQIC                         12625.551
                                    - 749                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    intercept   1125.9514    574.622      1.959      0.050      -0.286    2252.189
    ar.L1          0.9898      0.005    186.108      0.000       0.979       1.000
    ma.L1          0.0650      0.035      1.856      0.063      -0.004       0.134
    ma.L2         -0.0503      0.035     -1.428      0.153      -0.119       0.019
    sigma2      1.184e+06      0.433   2.73e+06      0.000    1.18e+06    1.18e+06
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):                28.07
    Prob(Q):                              0.92   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.38   Skew:                            -0.11
    Prob(H) (two-sided):                  0.00   Kurtosis:                         3.92
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    [2] Covariance matrix is singular or near-singular, with condition number 5.11e+21. Standard errors may be unstable.
    Shape dos dados de treino IBOV: (599, 1)
    Shape dos dados de teste IBOV: (150, 1)
    

    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_79_7.png)
    



    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_79_8.png)
    


#### Comparativo


```python
# Plotar o gráfico da previsão
plt.figure(figsize=(14, 7))

plt.plot(portfolio_return.index, portfolio_return, label='Histórico Portfólio', color='blue')
plt.plot(forecast_index, forecast_mean, label='Previsão Portfólio', color='blue', linestyle='--')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='lightsteelblue', alpha=0.3, label='Intervalo de Confiança Portfólio')

plt.plot(cdi_return.index, cdi_return, label='Histórico CDI', color='green')
plt.plot(forecast_index_cdi, forecast_mean_cdi, label='Previsão CDI', color='green',  linestyle='--')
plt.fill_between(forecast_index_cdi, forecast_conf_int_cdi.iloc[:, 0], forecast_conf_int_cdi.iloc[:, 1], color='darkseagreen', alpha=0.3, label='Intervalo de Confiança CDI')

plt.plot(ibov_return.index, ibov_return, label='Histórico IBOV', color='orange')
plt.plot(forecast_index_ibov, forecast_mean_ibov, label='Previsão IBOV', color='orange',  linestyle='--')
plt.fill_between(forecast_index_ibov, forecast_conf_int_ibov.iloc[:, 0], forecast_conf_int_ibov.iloc[:, 1], color='goldenrod', alpha=0.3, label='Intervalo de Confiança IBOV')

plt.xlabel('Data')
plt.ylabel('Retorno')
plt.title(f'Comparação entre Histórico e Previsão para Portfólio, CDI e IBOV ({best_window} dias)')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_81_0.png)
    


### Previsão do Portfólio - LSTM


```python
# Calcula o retorno acumulado do portfolio
portfolio_return = (historical_returns.cumsum() + 1) * portfolio_value

# Cria um DataFrame com o saldo do portfolio por dia
portfolio_balance = pd.DataFrame({
    'Saldo': portfolio_return
})

# Exibe o DataFrame
print(portfolio_balance)

```

                        Saldo
    Date                     
    2021-12-29   99071.921171
    2021-12-30   99367.847829
    2022-01-03   99440.143549
    2022-01-04   99392.851153
    2022-01-05   96943.004342
    ...                   ...
    2024-12-18  138793.858439
    2024-12-19  139955.749417
    2024-12-20  140883.137018
    2024-12-23  140507.538430
    2024-12-26  140650.629150
    
    [749 rows x 1 columns]
    


```python
# Separa linhas de treino e teste
qtd_linhas = len(portfolio_balance)
qtd_linhas_treino = round(qtd_linhas * 0.7)
qtd_linhas_teste = qtd_linhas - qtd_linhas_treino

info_treino = f'Quantidade de linhas de treino: {qtd_linhas_treino}'
info_teste = f'Quantidade de linhas de teste: {qtd_linhas_teste}'

print(info_treino)
print(info_teste)
```

    Quantidade de linhas de treino: 524
    Quantidade de linhas de teste: 225
    


```python
# Padroniza os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(portfolio_balance)
```


```python
# Separa os dados em treino e teste
train = df_scaled[:qtd_linhas_treino]
test = df_scaled[qtd_linhas_treino: qtd_linhas_treino + qtd_linhas_teste]

print(len(train) , len(test))
```

    524 225
    


```python
# Define numero de dias necessários para realizar a previsão do próximo dia
steps = 30

X_train, Y_train = create_df(train, steps )
X_test, Y_test = create_df(test, steps )

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
```

    (493, 30)
    (493,)
    (194, 30)
    (194,)
    


```python
# Gerando os dados esperados pelo modelo

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(X_train.shape)
print(X_test.shape)
```

    (493, 30, 1)
    (194, 30, 1)
    


```python
# Montando a rede

model = Sequential()
model.add(LSTM(35, return_sequences=True, input_shape=(steps, 1)))
model.add(LSTM(35, return_sequences=True))
model.add(LSTM(35))
model.add(Dropout(0,2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.summary()
```

    c:\Users\jeand\anaconda3\Lib\site-packages\keras\src\layers\rnn\rnn.py:204: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(**kwargs)
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_13"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ lstm_39 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">35</span>)         │         <span style="color: #00af00; text-decoration-color: #00af00">5,180</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_40 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">35</span>)         │         <span style="color: #00af00; text-decoration-color: #00af00">9,940</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_41 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">35</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">9,940</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_13 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">35</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_13 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">36</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">25,096</span> (98.03 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">25,096</span> (98.03 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
# Treinamento do Modelo

validation = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=steps, verbose=2)
```

    Epoch 1/100
    17/17 - 5s - 266ms/step - loss: 0.3435 - val_loss: 0.1777
    Epoch 2/100
    17/17 - 0s - 15ms/step - loss: 0.0774 - val_loss: 0.0925
    Epoch 3/100
    17/17 - 0s - 15ms/step - loss: 0.0592 - val_loss: 0.0298
    Epoch 4/100
    17/17 - 0s - 16ms/step - loss: 0.0536 - val_loss: 0.0346
    Epoch 5/100
    17/17 - 0s - 15ms/step - loss: 0.0483 - val_loss: 0.0348
    Epoch 6/100
    17/17 - 0s - 15ms/step - loss: 0.0448 - val_loss: 0.0331
    Epoch 7/100
    17/17 - 0s - 16ms/step - loss: 0.0440 - val_loss: 0.0262
    Epoch 8/100
    17/17 - 0s - 15ms/step - loss: 0.0421 - val_loss: 0.0317
    Epoch 9/100
    17/17 - 0s - 15ms/step - loss: 0.0379 - val_loss: 0.0310
    Epoch 10/100
    17/17 - 0s - 14ms/step - loss: 0.0345 - val_loss: 0.0251
    Epoch 11/100
    17/17 - 0s - 14ms/step - loss: 0.0345 - val_loss: 0.0252
    Epoch 12/100
    17/17 - 0s - 16ms/step - loss: 0.0324 - val_loss: 0.0284
    Epoch 13/100
    17/17 - 0s - 15ms/step - loss: 0.0299 - val_loss: 0.0249
    Epoch 14/100
    17/17 - 0s - 16ms/step - loss: 0.0290 - val_loss: 0.0279
    Epoch 15/100
    17/17 - 0s - 17ms/step - loss: 0.0281 - val_loss: 0.0235
    Epoch 16/100
    17/17 - 0s - 15ms/step - loss: 0.0294 - val_loss: 0.0264
    Epoch 17/100
    17/17 - 0s - 14ms/step - loss: 0.0260 - val_loss: 0.0268
    Epoch 18/100
    17/17 - 0s - 14ms/step - loss: 0.0257 - val_loss: 0.0242
    Epoch 19/100
    17/17 - 0s - 15ms/step - loss: 0.0253 - val_loss: 0.0294
    Epoch 20/100
    17/17 - 0s - 15ms/step - loss: 0.0232 - val_loss: 0.0274
    Epoch 21/100
    17/17 - 0s - 17ms/step - loss: 0.0226 - val_loss: 0.0198
    Epoch 22/100
    17/17 - 0s - 14ms/step - loss: 0.0217 - val_loss: 0.0215
    Epoch 23/100
    17/17 - 0s - 14ms/step - loss: 0.0217 - val_loss: 0.0222
    Epoch 24/100
    17/17 - 0s - 14ms/step - loss: 0.0209 - val_loss: 0.0199
    Epoch 25/100
    17/17 - 0s - 14ms/step - loss: 0.0204 - val_loss: 0.0226
    Epoch 26/100
    17/17 - 0s - 15ms/step - loss: 0.0207 - val_loss: 0.0166
    Epoch 27/100
    17/17 - 0s - 15ms/step - loss: 0.0197 - val_loss: 0.0182
    Epoch 28/100
    17/17 - 0s - 15ms/step - loss: 0.0206 - val_loss: 0.0164
    Epoch 29/100
    17/17 - 0s - 15ms/step - loss: 0.0176 - val_loss: 0.0152
    Epoch 30/100
    17/17 - 0s - 15ms/step - loss: 0.0164 - val_loss: 0.0145
    Epoch 31/100
    17/17 - 0s - 14ms/step - loss: 0.0160 - val_loss: 0.0150
    Epoch 32/100
    17/17 - 0s - 14ms/step - loss: 0.0152 - val_loss: 0.0141
    Epoch 33/100
    17/17 - 0s - 14ms/step - loss: 0.0147 - val_loss: 0.0174
    Epoch 34/100
    17/17 - 0s - 14ms/step - loss: 0.0142 - val_loss: 0.0165
    Epoch 35/100
    17/17 - 0s - 14ms/step - loss: 0.0138 - val_loss: 0.0127
    Epoch 36/100
    17/17 - 0s - 14ms/step - loss: 0.0129 - val_loss: 0.0122
    Epoch 37/100
    17/17 - 0s - 14ms/step - loss: 0.0135 - val_loss: 0.0117
    Epoch 38/100
    17/17 - 0s - 14ms/step - loss: 0.0128 - val_loss: 0.0118
    Epoch 39/100
    17/17 - 0s - 14ms/step - loss: 0.0136 - val_loss: 0.0114
    Epoch 40/100
    17/17 - 0s - 14ms/step - loss: 0.0123 - val_loss: 0.0107
    Epoch 41/100
    17/17 - 0s - 14ms/step - loss: 0.0112 - val_loss: 0.0124
    Epoch 42/100
    17/17 - 0s - 14ms/step - loss: 0.0108 - val_loss: 0.0113
    Epoch 43/100
    17/17 - 0s - 14ms/step - loss: 0.0102 - val_loss: 0.0155
    Epoch 44/100
    17/17 - 0s - 14ms/step - loss: 0.0097 - val_loss: 0.0213
    Epoch 45/100
    17/17 - 0s - 14ms/step - loss: 0.0107 - val_loss: 0.0194
    Epoch 46/100
    17/17 - 0s - 14ms/step - loss: 0.0101 - val_loss: 0.0121
    Epoch 47/100
    17/17 - 0s - 15ms/step - loss: 0.0100 - val_loss: 0.0183
    Epoch 48/100
    17/17 - 0s - 14ms/step - loss: 0.0100 - val_loss: 0.0126
    Epoch 49/100
    17/17 - 0s - 14ms/step - loss: 0.0100 - val_loss: 0.0084
    Epoch 50/100
    17/17 - 0s - 14ms/step - loss: 0.0118 - val_loss: 0.0086
    Epoch 51/100
    17/17 - 0s - 14ms/step - loss: 0.0106 - val_loss: 0.0086
    Epoch 52/100
    17/17 - 0s - 14ms/step - loss: 0.0090 - val_loss: 0.0089
    Epoch 53/100
    17/17 - 0s - 15ms/step - loss: 0.0087 - val_loss: 0.0071
    Epoch 54/100
    17/17 - 0s - 15ms/step - loss: 0.0093 - val_loss: 0.0079
    Epoch 55/100
    17/17 - 0s - 15ms/step - loss: 0.0086 - val_loss: 0.0071
    Epoch 56/100
    17/17 - 0s - 14ms/step - loss: 0.0083 - val_loss: 0.0125
    Epoch 57/100
    17/17 - 0s - 14ms/step - loss: 0.0094 - val_loss: 0.0132
    Epoch 58/100
    17/17 - 0s - 14ms/step - loss: 0.0111 - val_loss: 0.0135
    Epoch 59/100
    17/17 - 0s - 14ms/step - loss: 0.0092 - val_loss: 0.0082
    Epoch 60/100
    17/17 - 0s - 14ms/step - loss: 0.0089 - val_loss: 0.0167
    Epoch 61/100
    17/17 - 0s - 14ms/step - loss: 0.0088 - val_loss: 0.0164
    Epoch 62/100
    17/17 - 0s - 14ms/step - loss: 0.0084 - val_loss: 0.0097
    Epoch 63/100
    17/17 - 0s - 14ms/step - loss: 0.0085 - val_loss: 0.0065
    Epoch 64/100
    17/17 - 0s - 14ms/step - loss: 0.0083 - val_loss: 0.0063
    Epoch 65/100
    17/17 - 0s - 14ms/step - loss: 0.0090 - val_loss: 0.0090
    Epoch 66/100
    17/17 - 0s - 14ms/step - loss: 0.0100 - val_loss: 0.0083
    Epoch 67/100
    17/17 - 0s - 14ms/step - loss: 0.0090 - val_loss: 0.0086
    Epoch 68/100
    17/17 - 0s - 14ms/step - loss: 0.0087 - val_loss: 0.0064
    Epoch 69/100
    17/17 - 0s - 14ms/step - loss: 0.0079 - val_loss: 0.0078
    Epoch 70/100
    17/17 - 0s - 14ms/step - loss: 0.0079 - val_loss: 0.0077
    Epoch 71/100
    17/17 - 0s - 14ms/step - loss: 0.0077 - val_loss: 0.0083
    Epoch 72/100
    17/17 - 0s - 14ms/step - loss: 0.0078 - val_loss: 0.0071
    Epoch 73/100
    17/17 - 0s - 14ms/step - loss: 0.0081 - val_loss: 0.0101
    Epoch 74/100
    17/17 - 0s - 14ms/step - loss: 0.0077 - val_loss: 0.0064
    Epoch 75/100
    17/17 - 0s - 14ms/step - loss: 0.0083 - val_loss: 0.0099
    Epoch 76/100
    17/17 - 0s - 14ms/step - loss: 0.0076 - val_loss: 0.0081
    Epoch 77/100
    17/17 - 0s - 15ms/step - loss: 0.0076 - val_loss: 0.0059
    Epoch 78/100
    17/17 - 0s - 14ms/step - loss: 0.0080 - val_loss: 0.0137
    Epoch 79/100
    17/17 - 0s - 14ms/step - loss: 0.0091 - val_loss: 0.0126
    Epoch 80/100
    17/17 - 0s - 14ms/step - loss: 0.0079 - val_loss: 0.0076
    Epoch 81/100
    17/17 - 0s - 15ms/step - loss: 0.0081 - val_loss: 0.0105
    Epoch 82/100
    17/17 - 0s - 16ms/step - loss: 0.0076 - val_loss: 0.0062
    Epoch 83/100
    17/17 - 0s - 18ms/step - loss: 0.0085 - val_loss: 0.0061
    Epoch 84/100
    17/17 - 0s - 18ms/step - loss: 0.0085 - val_loss: 0.0074
    Epoch 85/100
    17/17 - 0s - 14ms/step - loss: 0.0076 - val_loss: 0.0079
    Epoch 86/100
    17/17 - 0s - 14ms/step - loss: 0.0080 - val_loss: 0.0090
    Epoch 87/100
    17/17 - 0s - 14ms/step - loss: 0.0079 - val_loss: 0.0172
    Epoch 88/100
    17/17 - 0s - 15ms/step - loss: 0.0081 - val_loss: 0.0096
    Epoch 89/100
    17/17 - 0s - 14ms/step - loss: 0.0076 - val_loss: 0.0069
    Epoch 90/100
    17/17 - 0s - 14ms/step - loss: 0.0077 - val_loss: 0.0142
    Epoch 91/100
    17/17 - 0s - 18ms/step - loss: 0.0076 - val_loss: 0.0079
    Epoch 92/100
    17/17 - 0s - 18ms/step - loss: 0.0079 - val_loss: 0.0086
    Epoch 93/100
    17/17 - 0s - 17ms/step - loss: 0.0074 - val_loss: 0.0060
    Epoch 94/100
    17/17 - 0s - 15ms/step - loss: 0.0077 - val_loss: 0.0060
    Epoch 95/100
    17/17 - 0s - 15ms/step - loss: 0.0075 - val_loss: 0.0096
    Epoch 96/100
    17/17 - 0s - 14ms/step - loss: 0.0080 - val_loss: 0.0066
    Epoch 97/100
    17/17 - 0s - 14ms/step - loss: 0.0086 - val_loss: 0.0066
    Epoch 98/100
    17/17 - 0s - 14ms/step - loss: 0.0076 - val_loss: 0.0063
    Epoch 99/100
    17/17 - 0s - 14ms/step - loss: 0.0076 - val_loss: 0.0057
    Epoch 100/100
    17/17 - 0s - 18ms/step - loss: 0.0083 - val_loss: 0.0089
    


```python
# Estudo da validação
plt.plot(validation.history['loss'], label='Training Loss')
plt.plot(validation.history['val_loss'], label='Validation Loss')
plt.title('Erro do Modelo')
plt.ylabel('Erro')
plt.xlabel('Época')
plt.legend()
plt.show()
```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_91_0.png)
    



```python
# Realiza a previsão

prev = model.predict(X_test)
prev = scaler.inverse_transform(prev)

len_test = len(test)
len_prev = len(prev)

print(len_test, len_prev)

days_input_steps = len_test - steps
input_steps = test[days_input_steps:]
input_steps = np.array(input_steps).reshape(1, -1)
input_steps.shape

# Transformar em lista

list_output_steps = list(input_steps)
list_output_steps = list_output_steps[0].tolist()
list_output_steps
```

    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 65ms/step
    225 194
    




    [1.1781678324878473,
     1.2944636196136776,
     1.2587324704908012,
     1.2734747765126022,
     1.2715791228995739,
     1.3188499331946657,
     1.2691500962066078,
     1.375806189945475,
     1.3414912819256135,
     1.3816506478872126,
     1.2551188357335037,
     1.0491109314362885,
     1.0216558690009847,
     0.9590693828571786,
     1.0166488830552007,
     0.9346495266771316,
     1.0442073887586663,
     0.8545627296661267,
     0.9112853666881668,
     0.983321728449009,
     1.0369357137434054,
     0.8343651724307409,
     0.7215209287801967,
     0.6292616295185279,
     0.7000937051065068,
     0.516477898600993,
     0.5945763667539274,
     0.6569122935892208,
     0.6316657989240794,
     0.6412838837904699]




```python
# loop para prever os proximos dias

pred_output = []
i = 0
n_future = best_window
while(i<n_future):
    if(len(list_output_steps)>steps):
        input_steps = np.array(list_output_steps[1:])
        print(f"{i} dia. Valores de Entrada -> {input_steps}")
        input_steps = input_steps.reshape(1, -1)
        input_steps = input_steps.reshape((1, steps, 1))
        pred = model.predict(input_steps, verbose=0)
        print(f"{i} dia. Valor Previsto -> {pred}")
        list_output_steps.extend(pred[0].tolist())
        list_output_steps = list_output_steps[1:]
        pred_output.extend(pred.tolist())
        i=i+1
    else:
        input_steps = input_steps.reshape((1, steps, 1))
        pred = model.predict(input_steps, verbose=0)
        print(pred[0])
        list_output_steps.extend(pred[0].tolist())
        print(len(list_output_steps))
        pred_output.extend(pred.tolist())
        i=i+1
print(f'Previsões -> {pred_output}')
```

    [0.7527911]
    31
    1 dia. Valores de Entrada -> [1.29446362 1.25873247 1.27347478 1.27157912 1.31884993 1.2691501
     1.37580619 1.34149128 1.38165065 1.25511884 1.04911093 1.02165587
     0.95906938 1.01664888 0.93464953 1.04420739 0.85456273 0.91128537
     0.98332173 1.03693571 0.83436517 0.72152093 0.62926163 0.70009371
     0.5164779  0.59457637 0.65691229 0.6316658  0.64128388 0.75279111]
    1 dia. Valor Previsto -> [[0.8465024]]
    2 dia. Valores de Entrada -> [1.25873247 1.27347478 1.27157912 1.31884993 1.2691501  1.37580619
     1.34149128 1.38165065 1.25511884 1.04911093 1.02165587 0.95906938
     1.01664888 0.93464953 1.04420739 0.85456273 0.91128537 0.98332173
     1.03693571 0.83436517 0.72152093 0.62926163 0.70009371 0.5164779
     0.59457637 0.65691229 0.6316658  0.64128388 0.75279111 0.84650242]
    2 dia. Valor Previsto -> [[0.9299151]]
    3 dia. Valores de Entrada -> [1.27347478 1.27157912 1.31884993 1.2691501  1.37580619 1.34149128
     1.38165065 1.25511884 1.04911093 1.02165587 0.95906938 1.01664888
     0.93464953 1.04420739 0.85456273 0.91128537 0.98332173 1.03693571
     0.83436517 0.72152093 0.62926163 0.70009371 0.5164779  0.59457637
     0.65691229 0.6316658  0.64128388 0.75279111 0.84650242 0.92991507]
    3 dia. Valor Previsto -> [[1.0117124]]
    4 dia. Valores de Entrada -> [1.27157912 1.31884993 1.2691501  1.37580619 1.34149128 1.38165065
     1.25511884 1.04911093 1.02165587 0.95906938 1.01664888 0.93464953
     1.04420739 0.85456273 0.91128537 0.98332173 1.03693571 0.83436517
     0.72152093 0.62926163 0.70009371 0.5164779  0.59457637 0.65691229
     0.6316658  0.64128388 0.75279111 0.84650242 0.92991507 1.01171243]
    4 dia. Valor Previsto -> [[1.0920227]]
    5 dia. Valores de Entrada -> [1.31884993 1.2691501  1.37580619 1.34149128 1.38165065 1.25511884
     1.04911093 1.02165587 0.95906938 1.01664888 0.93464953 1.04420739
     0.85456273 0.91128537 0.98332173 1.03693571 0.83436517 0.72152093
     0.62926163 0.70009371 0.5164779  0.59457637 0.65691229 0.6316658
     0.64128388 0.75279111 0.84650242 0.92991507 1.01171243 1.09202266]
    5 dia. Valor Previsto -> [[1.1687006]]
    6 dia. Valores de Entrada -> [1.2691501  1.37580619 1.34149128 1.38165065 1.25511884 1.04911093
     1.02165587 0.95906938 1.01664888 0.93464953 1.04420739 0.85456273
     0.91128537 0.98332173 1.03693571 0.83436517 0.72152093 0.62926163
     0.70009371 0.5164779  0.59457637 0.65691229 0.6316658  0.64128388
     0.75279111 0.84650242 0.92991507 1.01171243 1.09202266 1.16870058]
    6 dia. Valor Previsto -> [[1.2388437]]
    7 dia. Valores de Entrada -> [1.37580619 1.34149128 1.38165065 1.25511884 1.04911093 1.02165587
     0.95906938 1.01664888 0.93464953 1.04420739 0.85456273 0.91128537
     0.98332173 1.03693571 0.83436517 0.72152093 0.62926163 0.70009371
     0.5164779  0.59457637 0.65691229 0.6316658  0.64128388 0.75279111
     0.84650242 0.92991507 1.01171243 1.09202266 1.16870058 1.23884368]
    7 dia. Valor Previsto -> [[1.3000295]]
    8 dia. Valores de Entrada -> [1.34149128 1.38165065 1.25511884 1.04911093 1.02165587 0.95906938
     1.01664888 0.93464953 1.04420739 0.85456273 0.91128537 0.98332173
     1.03693571 0.83436517 0.72152093 0.62926163 0.70009371 0.5164779
     0.59457637 0.65691229 0.6316658  0.64128388 0.75279111 0.84650242
     0.92991507 1.01171243 1.09202266 1.16870058 1.23884368 1.30002952]
    8 dia. Valor Previsto -> [[1.3505486]]
    9 dia. Valores de Entrada -> [1.38165065 1.25511884 1.04911093 1.02165587 0.95906938 1.01664888
     0.93464953 1.04420739 0.85456273 0.91128537 0.98332173 1.03693571
     0.83436517 0.72152093 0.62926163 0.70009371 0.5164779  0.59457637
     0.65691229 0.6316658  0.64128388 0.75279111 0.84650242 0.92991507
     1.01171243 1.09202266 1.16870058 1.23884368 1.30002952 1.35054862]
    9 dia. Valor Previsto -> [[1.3899909]]
    10 dia. Valores de Entrada -> [1.25511884 1.04911093 1.02165587 0.95906938 1.01664888 0.93464953
     1.04420739 0.85456273 0.91128537 0.98332173 1.03693571 0.83436517
     0.72152093 0.62926163 0.70009371 0.5164779  0.59457637 0.65691229
     0.6316658  0.64128388 0.75279111 0.84650242 0.92991507 1.01171243
     1.09202266 1.16870058 1.23884368 1.30002952 1.35054862 1.38999093]
    10 dia. Valor Previsto -> [[1.418569]]
    11 dia. Valores de Entrada -> [1.04911093 1.02165587 0.95906938 1.01664888 0.93464953 1.04420739
     0.85456273 0.91128537 0.98332173 1.03693571 0.83436517 0.72152093
     0.62926163 0.70009371 0.5164779  0.59457637 0.65691229 0.6316658
     0.64128388 0.75279111 0.84650242 0.92991507 1.01171243 1.09202266
     1.16870058 1.23884368 1.30002952 1.35054862 1.38999093 1.41856897]
    11 dia. Valor Previsto -> [[1.4373146]]
    12 dia. Valores de Entrada -> [1.02165587 0.95906938 1.01664888 0.93464953 1.04420739 0.85456273
     0.91128537 0.98332173 1.03693571 0.83436517 0.72152093 0.62926163
     0.70009371 0.5164779  0.59457637 0.65691229 0.6316658  0.64128388
     0.75279111 0.84650242 0.92991507 1.01171243 1.09202266 1.16870058
     1.23884368 1.30002952 1.35054862 1.38999093 1.41856897 1.43731463]
    12 dia. Valor Previsto -> [[1.4474936]]
    13 dia. Valores de Entrada -> [0.95906938 1.01664888 0.93464953 1.04420739 0.85456273 0.91128537
     0.98332173 1.03693571 0.83436517 0.72152093 0.62926163 0.70009371
     0.5164779  0.59457637 0.65691229 0.6316658  0.64128388 0.75279111
     0.84650242 0.92991507 1.01171243 1.09202266 1.16870058 1.23884368
     1.30002952 1.35054862 1.38999093 1.41856897 1.43731463 1.44749355]
    13 dia. Valor Previsto -> [[1.4501573]]
    14 dia. Valores de Entrada -> [1.01664888 0.93464953 1.04420739 0.85456273 0.91128537 0.98332173
     1.03693571 0.83436517 0.72152093 0.62926163 0.70009371 0.5164779
     0.59457637 0.65691229 0.6316658  0.64128388 0.75279111 0.84650242
     0.92991507 1.01171243 1.09202266 1.16870058 1.23884368 1.30002952
     1.35054862 1.38999093 1.41856897 1.43731463 1.44749355 1.45015728]
    14 dia. Valor Previsto -> [[1.4466456]]
    15 dia. Valores de Entrada -> [0.93464953 1.04420739 0.85456273 0.91128537 0.98332173 1.03693571
     0.83436517 0.72152093 0.62926163 0.70009371 0.5164779  0.59457637
     0.65691229 0.6316658  0.64128388 0.75279111 0.84650242 0.92991507
     1.01171243 1.09202266 1.16870058 1.23884368 1.30002952 1.35054862
     1.38999093 1.41856897 1.43731463 1.44749355 1.45015728 1.44664562]
    15 dia. Valor Previsto -> [[1.4379851]]
    16 dia. Valores de Entrada -> [1.04420739 0.85456273 0.91128537 0.98332173 1.03693571 0.83436517
     0.72152093 0.62926163 0.70009371 0.5164779  0.59457637 0.65691229
     0.6316658  0.64128388 0.75279111 0.84650242 0.92991507 1.01171243
     1.09202266 1.16870058 1.23884368 1.30002952 1.35054862 1.38999093
     1.41856897 1.43731463 1.44749355 1.45015728 1.44664562 1.43798506]
    16 dia. Valor Previsto -> [[1.4254904]]
    17 dia. Valores de Entrada -> [0.85456273 0.91128537 0.98332173 1.03693571 0.83436517 0.72152093
     0.62926163 0.70009371 0.5164779  0.59457637 0.65691229 0.6316658
     0.64128388 0.75279111 0.84650242 0.92991507 1.01171243 1.09202266
     1.16870058 1.23884368 1.30002952 1.35054862 1.38999093 1.41856897
     1.43731463 1.44749355 1.45015728 1.44664562 1.43798506 1.42549038]
    17 dia. Valor Previsto -> [[1.4098371]]
    18 dia. Valores de Entrada -> [0.91128537 0.98332173 1.03693571 0.83436517 0.72152093 0.62926163
     0.70009371 0.5164779  0.59457637 0.65691229 0.6316658  0.64128388
     0.75279111 0.84650242 0.92991507 1.01171243 1.09202266 1.16870058
     1.23884368 1.30002952 1.35054862 1.38999093 1.41856897 1.43731463
     1.44749355 1.45015728 1.44664562 1.43798506 1.42549038 1.40983713]
    18 dia. Valor Previsto -> [[1.3923908]]
    19 dia. Valores de Entrada -> [0.98332173 1.03693571 0.83436517 0.72152093 0.62926163 0.70009371
     0.5164779  0.59457637 0.65691229 0.6316658  0.64128388 0.75279111
     0.84650242 0.92991507 1.01171243 1.09202266 1.16870058 1.23884368
     1.30002952 1.35054862 1.38999093 1.41856897 1.43731463 1.44749355
     1.45015728 1.44664562 1.43798506 1.42549038 1.40983713 1.39239085]
    19 dia. Valor Previsto -> [[1.3735653]]
    20 dia. Valores de Entrada -> [1.03693571 0.83436517 0.72152093 0.62926163 0.70009371 0.5164779
     0.59457637 0.65691229 0.6316658  0.64128388 0.75279111 0.84650242
     0.92991507 1.01171243 1.09202266 1.16870058 1.23884368 1.30002952
     1.35054862 1.38999093 1.41856897 1.43731463 1.44749355 1.45015728
     1.44664562 1.43798506 1.42549038 1.40983713 1.39239085 1.37356532]
    20 dia. Valor Previsto -> [[1.3540322]]
    21 dia. Valores de Entrada -> [0.83436517 0.72152093 0.62926163 0.70009371 0.5164779  0.59457637
     0.65691229 0.6316658  0.64128388 0.75279111 0.84650242 0.92991507
     1.01171243 1.09202266 1.16870058 1.23884368 1.30002952 1.35054862
     1.38999093 1.41856897 1.43731463 1.44749355 1.45015728 1.44664562
     1.43798506 1.42549038 1.40983713 1.39239085 1.37356532 1.35403216]
    21 dia. Valor Previsto -> [[1.3344507]]
    22 dia. Valores de Entrada -> [0.72152093 0.62926163 0.70009371 0.5164779  0.59457637 0.65691229
     0.6316658  0.64128388 0.75279111 0.84650242 0.92991507 1.01171243
     1.09202266 1.16870058 1.23884368 1.30002952 1.35054862 1.38999093
     1.41856897 1.43731463 1.44749355 1.45015728 1.44664562 1.43798506
     1.42549038 1.40983713 1.39239085 1.37356532 1.35403216 1.33445072]
    22 dia. Valor Previsto -> [[1.3158484]]
    23 dia. Valores de Entrada -> [0.62926163 0.70009371 0.5164779  0.59457637 0.65691229 0.6316658
     0.64128388 0.75279111 0.84650242 0.92991507 1.01171243 1.09202266
     1.16870058 1.23884368 1.30002952 1.35054862 1.38999093 1.41856897
     1.43731463 1.44749355 1.45015728 1.44664562 1.43798506 1.42549038
     1.40983713 1.39239085 1.37356532 1.35403216 1.33445072 1.31584835]
    23 dia. Valor Previsto -> [[1.298707]]
    24 dia. Valores de Entrada -> [0.70009371 0.5164779  0.59457637 0.65691229 0.6316658  0.64128388
     0.75279111 0.84650242 0.92991507 1.01171243 1.09202266 1.16870058
     1.23884368 1.30002952 1.35054862 1.38999093 1.41856897 1.43731463
     1.44749355 1.45015728 1.44664562 1.43798506 1.42549038 1.40983713
     1.39239085 1.37356532 1.35403216 1.33445072 1.31584835 1.29870701]
    24 dia. Valor Previsto -> [[1.2835253]]
    25 dia. Valores de Entrada -> [0.5164779  0.59457637 0.65691229 0.6316658  0.64128388 0.75279111
     0.84650242 0.92991507 1.01171243 1.09202266 1.16870058 1.23884368
     1.30002952 1.35054862 1.38999093 1.41856897 1.43731463 1.44749355
     1.45015728 1.44664562 1.43798506 1.42549038 1.40983713 1.39239085
     1.37356532 1.35403216 1.33445072 1.31584835 1.29870701 1.28352535]
    25 dia. Valor Previsto -> [[1.2704431]]
    26 dia. Valores de Entrada -> [0.59457637 0.65691229 0.6316658  0.64128388 0.75279111 0.84650242
     0.92991507 1.01171243 1.09202266 1.16870058 1.23884368 1.30002952
     1.35054862 1.38999093 1.41856897 1.43731463 1.44749355 1.45015728
     1.44664562 1.43798506 1.42549038 1.40983713 1.39239085 1.37356532
     1.35403216 1.33445072 1.31584835 1.29870701 1.28352535 1.27044308]
    26 dia. Valor Previsto -> [[1.2602203]]
    27 dia. Valores de Entrada -> [0.65691229 0.6316658  0.64128388 0.75279111 0.84650242 0.92991507
     1.01171243 1.09202266 1.16870058 1.23884368 1.30002952 1.35054862
     1.38999093 1.41856897 1.43731463 1.44749355 1.45015728 1.44664562
     1.43798506 1.42549038 1.40983713 1.39239085 1.37356532 1.35403216
     1.33445072 1.31584835 1.29870701 1.28352535 1.27044308 1.26022029]
    27 dia. Valor Previsto -> [[1.2527194]]
    28 dia. Valores de Entrada -> [0.6316658  0.64128388 0.75279111 0.84650242 0.92991507 1.01171243
     1.09202266 1.16870058 1.23884368 1.30002952 1.35054862 1.38999093
     1.41856897 1.43731463 1.44749355 1.45015728 1.44664562 1.43798506
     1.42549038 1.40983713 1.39239085 1.37356532 1.35403216 1.33445072
     1.31584835 1.29870701 1.28352535 1.27044308 1.26022029 1.2527194 ]
    28 dia. Valor Previsto -> [[1.2480686]]
    29 dia. Valores de Entrada -> [0.64128388 0.75279111 0.84650242 0.92991507 1.01171243 1.09202266
     1.16870058 1.23884368 1.30002952 1.35054862 1.38999093 1.41856897
     1.43731463 1.44749355 1.45015728 1.44664562 1.43798506 1.42549038
     1.40983713 1.39239085 1.37356532 1.35403216 1.33445072 1.31584835
     1.29870701 1.28352535 1.27044308 1.26022029 1.2527194  1.24806857]
    29 dia. Valor Previsto -> [[1.2464757]]
    30 dia. Valores de Entrada -> [0.75279111 0.84650242 0.92991507 1.01171243 1.09202266 1.16870058
     1.23884368 1.30002952 1.35054862 1.38999093 1.41856897 1.43731463
     1.44749355 1.45015728 1.44664562 1.43798506 1.42549038 1.40983713
     1.39239085 1.37356532 1.35403216 1.33445072 1.31584835 1.29870701
     1.28352535 1.27044308 1.26022029 1.2527194  1.24806857 1.2464757 ]
    30 dia. Valor Previsto -> [[1.2478862]]
    31 dia. Valores de Entrada -> [0.84650242 0.92991507 1.01171243 1.09202266 1.16870058 1.23884368
     1.30002952 1.35054862 1.38999093 1.41856897 1.43731463 1.44749355
     1.45015728 1.44664562 1.43798506 1.42549038 1.40983713 1.39239085
     1.37356532 1.35403216 1.33445072 1.31584835 1.29870701 1.28352535
     1.27044308 1.26022029 1.2527194  1.24806857 1.2464757  1.24788618]
    31 dia. Valor Previsto -> [[1.2519729]]
    32 dia. Valores de Entrada -> [0.92991507 1.01171243 1.09202266 1.16870058 1.23884368 1.30002952
     1.35054862 1.38999093 1.41856897 1.43731463 1.44749355 1.45015728
     1.44664562 1.43798506 1.42549038 1.40983713 1.39239085 1.37356532
     1.35403216 1.33445072 1.31584835 1.29870701 1.28352535 1.27044308
     1.26022029 1.2527194  1.24806857 1.2464757  1.24788618 1.25197291]
    32 dia. Valor Previsto -> [[1.2584738]]
    33 dia. Valores de Entrada -> [1.01171243 1.09202266 1.16870058 1.23884368 1.30002952 1.35054862
     1.38999093 1.41856897 1.43731463 1.44749355 1.45015728 1.44664562
     1.43798506 1.42549038 1.40983713 1.39239085 1.37356532 1.35403216
     1.33445072 1.31584835 1.29870701 1.28352535 1.27044308 1.26022029
     1.2527194  1.24806857 1.2464757  1.24788618 1.25197291 1.25847375]
    33 dia. Valor Previsto -> [[1.2670503]]
    34 dia. Valores de Entrada -> [1.09202266 1.16870058 1.23884368 1.30002952 1.35054862 1.38999093
     1.41856897 1.43731463 1.44749355 1.45015728 1.44664562 1.43798506
     1.42549038 1.40983713 1.39239085 1.37356532 1.35403216 1.33445072
     1.31584835 1.29870701 1.28352535 1.27044308 1.26022029 1.2527194
     1.24806857 1.2464757  1.24788618 1.25197291 1.25847375 1.26705027]
    34 dia. Valor Previsto -> [[1.2772893]]
    35 dia. Valores de Entrada -> [1.16870058 1.23884368 1.30002952 1.35054862 1.38999093 1.41856897
     1.43731463 1.44749355 1.45015728 1.44664562 1.43798506 1.42549038
     1.40983713 1.39239085 1.37356532 1.35403216 1.33445072 1.31584835
     1.29870701 1.28352535 1.27044308 1.26022029 1.2527194  1.24806857
     1.2464757  1.24788618 1.25197291 1.25847375 1.26705027 1.27728927]
    35 dia. Valor Previsto -> [[1.2887244]]
    36 dia. Valores de Entrada -> [1.23884368 1.30002952 1.35054862 1.38999093 1.41856897 1.43731463
     1.44749355 1.45015728 1.44664562 1.43798506 1.42549038 1.40983713
     1.39239085 1.37356532 1.35403216 1.33445072 1.31584835 1.29870701
     1.28352535 1.27044308 1.26022029 1.2527194  1.24806857 1.2464757
     1.24788618 1.25197291 1.25847375 1.26705027 1.27728927 1.28872442]
    36 dia. Valor Previsto -> [[1.3008631]]
    37 dia. Valores de Entrada -> [1.30002952 1.35054862 1.38999093 1.41856897 1.43731463 1.44749355
     1.45015728 1.44664562 1.43798506 1.42549038 1.40983713 1.39239085
     1.37356532 1.35403216 1.33445072 1.31584835 1.29870701 1.28352535
     1.27044308 1.26022029 1.2527194  1.24806857 1.2464757  1.24788618
     1.25197291 1.25847375 1.26705027 1.27728927 1.28872442 1.30086315]
    37 dia. Valor Previsto -> [[1.3132102]]
    38 dia. Valores de Entrada -> [1.35054862 1.38999093 1.41856897 1.43731463 1.44749355 1.45015728
     1.44664562 1.43798506 1.42549038 1.40983713 1.39239085 1.37356532
     1.35403216 1.33445072 1.31584835 1.29870701 1.28352535 1.27044308
     1.26022029 1.2527194  1.24806857 1.2464757  1.24788618 1.25197291
     1.25847375 1.26705027 1.27728927 1.28872442 1.30086315 1.31321025]
    38 dia. Valor Previsto -> [[1.3252949]]
    39 dia. Valores de Entrada -> [1.38999093 1.41856897 1.43731463 1.44749355 1.45015728 1.44664562
     1.43798506 1.42549038 1.40983713 1.39239085 1.37356532 1.35403216
     1.33445072 1.31584835 1.29870701 1.28352535 1.27044308 1.26022029
     1.2527194  1.24806857 1.2464757  1.24788618 1.25197291 1.25847375
     1.26705027 1.27728927 1.28872442 1.30086315 1.31321025 1.32529485]
    39 dia. Valor Previsto -> [[1.3366903]]
    40 dia. Valores de Entrada -> [1.41856897 1.43731463 1.44749355 1.45015728 1.44664562 1.43798506
     1.42549038 1.40983713 1.39239085 1.37356532 1.35403216 1.33445072
     1.31584835 1.29870701 1.28352535 1.27044308 1.26022029 1.2527194
     1.24806857 1.2464757  1.24788618 1.25197291 1.25847375 1.26705027
     1.27728927 1.28872442 1.30086315 1.31321025 1.32529485 1.33669031]
    40 dia. Valor Previsto -> [[1.3470335]]
    41 dia. Valores de Entrada -> [1.43731463 1.44749355 1.45015728 1.44664562 1.43798506 1.42549038
     1.40983713 1.39239085 1.37356532 1.35403216 1.33445072 1.31584835
     1.29870701 1.28352535 1.27044308 1.26022029 1.2527194  1.24806857
     1.2464757  1.24788618 1.25197291 1.25847375 1.26705027 1.27728927
     1.28872442 1.30086315 1.31321025 1.32529485 1.33669031 1.3470335 ]
    41 dia. Valor Previsto -> [[1.3560373]]
    42 dia. Valores de Entrada -> [1.44749355 1.45015728 1.44664562 1.43798506 1.42549038 1.40983713
     1.39239085 1.37356532 1.35403216 1.33445072 1.31584835 1.29870701
     1.28352535 1.27044308 1.26022029 1.2527194  1.24806857 1.2464757
     1.24788618 1.25197291 1.25847375 1.26705027 1.27728927 1.28872442
     1.30086315 1.31321025 1.32529485 1.33669031 1.3470335  1.35603726]
    42 dia. Valor Previsto -> [[1.3634925]]
    43 dia. Valores de Entrada -> [1.45015728 1.44664562 1.43798506 1.42549038 1.40983713 1.39239085
     1.37356532 1.35403216 1.33445072 1.31584835 1.29870701 1.28352535
     1.27044308 1.26022029 1.2527194  1.24806857 1.2464757  1.24788618
     1.25197291 1.25847375 1.26705027 1.27728927 1.28872442 1.30086315
     1.31321025 1.32529485 1.33669031 1.3470335  1.35603726 1.36349249]
    43 dia. Valor Previsto -> [[1.369272]]
    44 dia. Valores de Entrada -> [1.44664562 1.43798506 1.42549038 1.40983713 1.39239085 1.37356532
     1.35403216 1.33445072 1.31584835 1.29870701 1.28352535 1.27044308
     1.26022029 1.2527194  1.24806857 1.2464757  1.24788618 1.25197291
     1.25847375 1.26705027 1.27728927 1.28872442 1.30086315 1.31321025
     1.32529485 1.33669031 1.3470335  1.35603726 1.36349249 1.36927199]
    44 dia. Valor Previsto -> [[1.3733267]]
    45 dia. Valores de Entrada -> [1.43798506 1.42549038 1.40983713 1.39239085 1.37356532 1.35403216
     1.33445072 1.31584835 1.29870701 1.28352535 1.27044308 1.26022029
     1.2527194  1.24806857 1.2464757  1.24788618 1.25197291 1.25847375
     1.26705027 1.27728927 1.28872442 1.30086315 1.31321025 1.32529485
     1.33669031 1.3470335  1.35603726 1.36349249 1.36927199 1.37332666]
    45 dia. Valor Previsto -> [[1.3756746]]
    46 dia. Valores de Entrada -> [1.42549038 1.40983713 1.39239085 1.37356532 1.35403216 1.33445072
     1.31584835 1.29870701 1.28352535 1.27044308 1.26022029 1.2527194
     1.24806857 1.2464757  1.24788618 1.25197291 1.25847375 1.26705027
     1.27728927 1.28872442 1.30086315 1.31321025 1.32529485 1.33669031
     1.3470335  1.35603726 1.36349249 1.36927199 1.37332666 1.37567461]
    46 dia. Valor Previsto -> [[1.3763964]]
    47 dia. Valores de Entrada -> [1.40983713 1.39239085 1.37356532 1.35403216 1.33445072 1.31584835
     1.29870701 1.28352535 1.27044308 1.26022029 1.2527194  1.24806857
     1.2464757  1.24788618 1.25197291 1.25847375 1.26705027 1.27728927
     1.28872442 1.30086315 1.31321025 1.32529485 1.33669031 1.3470335
     1.35603726 1.36349249 1.36927199 1.37332666 1.37567461 1.37639642]
    47 dia. Valor Previsto -> [[1.3756183]]
    48 dia. Valores de Entrada -> [1.39239085 1.37356532 1.35403216 1.33445072 1.31584835 1.29870701
     1.28352535 1.27044308 1.26022029 1.2527194  1.24806857 1.2464757
     1.24788618 1.25197291 1.25847375 1.26705027 1.27728927 1.28872442
     1.30086315 1.31321025 1.32529485 1.33669031 1.3470335  1.35603726
     1.36349249 1.36927199 1.37332666 1.37567461 1.37639642 1.37561834]
    48 dia. Valor Previsto -> [[1.3735056]]
    49 dia. Valores de Entrada -> [1.37356532 1.35403216 1.33445072 1.31584835 1.29870701 1.28352535
     1.27044308 1.26022029 1.2527194  1.24806857 1.2464757  1.24788618
     1.25197291 1.25847375 1.26705027 1.27728927 1.28872442 1.30086315
     1.31321025 1.32529485 1.33669031 1.3470335  1.35603726 1.36349249
     1.36927199 1.37332666 1.37567461 1.37639642 1.37561834 1.37350559]
    49 dia. Valor Previsto -> [[1.3702503]]
    50 dia. Valores de Entrada -> [1.35403216 1.33445072 1.31584835 1.29870701 1.28352535 1.27044308
     1.26022029 1.2527194  1.24806857 1.2464757  1.24788618 1.25197291
     1.25847375 1.26705027 1.27728927 1.28872442 1.30086315 1.31321025
     1.32529485 1.33669031 1.3470335  1.35603726 1.36349249 1.36927199
     1.37332666 1.37567461 1.37639642 1.37561834 1.37350559 1.37025034]
    50 dia. Valor Previsto -> [[1.3660618]]
    51 dia. Valores de Entrada -> [1.33445072 1.31584835 1.29870701 1.28352535 1.27044308 1.26022029
     1.2527194  1.24806857 1.2464757  1.24788618 1.25197291 1.25847375
     1.26705027 1.27728927 1.28872442 1.30086315 1.31321025 1.32529485
     1.33669031 1.3470335  1.35603726 1.36349249 1.36927199 1.37332666
     1.37567461 1.37639642 1.37561834 1.37350559 1.37025034 1.36606181]
    51 dia. Valor Previsto -> [[1.3611583]]
    52 dia. Valores de Entrada -> [1.31584835 1.29870701 1.28352535 1.27044308 1.26022029 1.2527194
     1.24806857 1.2464757  1.24788618 1.25197291 1.25847375 1.26705027
     1.27728927 1.28872442 1.30086315 1.31321025 1.32529485 1.33669031
     1.3470335  1.35603726 1.36349249 1.36927199 1.37332666 1.37567461
     1.37639642 1.37561834 1.37350559 1.37025034 1.36606181 1.36115825]
    52 dia. Valor Previsto -> [[1.3557601]]
    53 dia. Valores de Entrada -> [1.29870701 1.28352535 1.27044308 1.26022029 1.2527194  1.24806857
     1.2464757  1.24788618 1.25197291 1.25847375 1.26705027 1.27728927
     1.28872442 1.30086315 1.31321025 1.32529485 1.33669031 1.3470335
     1.35603726 1.36349249 1.36927199 1.37332666 1.37567461 1.37639642
     1.37561834 1.37350559 1.37025034 1.36606181 1.36115825 1.3557601 ]
    53 dia. Valor Previsto -> [[1.350083]]
    54 dia. Valores de Entrada -> [1.28352535 1.27044308 1.26022029 1.2527194  1.24806857 1.2464757
     1.24788618 1.25197291 1.25847375 1.26705027 1.27728927 1.28872442
     1.30086315 1.31321025 1.32529485 1.33669031 1.3470335  1.35603726
     1.36349249 1.36927199 1.37332666 1.37567461 1.37639642 1.37561834
     1.37350559 1.37025034 1.36606181 1.36115825 1.3557601  1.35008299]
    54 dia. Valor Previsto -> [[1.3443314]]
    55 dia. Valores de Entrada -> [1.27044308 1.26022029 1.2527194  1.24806857 1.2464757  1.24788618
     1.25197291 1.25847375 1.26705027 1.27728927 1.28872442 1.30086315
     1.31321025 1.32529485 1.33669031 1.3470335  1.35603726 1.36349249
     1.36927199 1.37332666 1.37567461 1.37639642 1.37561834 1.37350559
     1.37025034 1.36606181 1.36115825 1.3557601  1.35008299 1.34433138]
    55 dia. Valor Previsto -> [[1.3386956]]
    56 dia. Valores de Entrada -> [1.26022029 1.2527194  1.24806857 1.2464757  1.24788618 1.25197291
     1.25847375 1.26705027 1.27728927 1.28872442 1.30086315 1.31321025
     1.32529485 1.33669031 1.3470335  1.35603726 1.36349249 1.36927199
     1.37332666 1.37567461 1.37639642 1.37561834 1.37350559 1.37025034
     1.36606181 1.36115825 1.3557601  1.35008299 1.34433138 1.33869565]
    56 dia. Valor Previsto -> [[1.3333482]]
    57 dia. Valores de Entrada -> [1.2527194  1.24806857 1.2464757  1.24788618 1.25197291 1.25847375
     1.26705027 1.27728927 1.28872442 1.30086315 1.31321025 1.32529485
     1.33669031 1.3470335  1.35603726 1.36349249 1.36927199 1.37332666
     1.37567461 1.37639642 1.37561834 1.37350559 1.37025034 1.36606181
     1.36115825 1.3557601  1.35008299 1.34433138 1.33869565 1.33334816]
    57 dia. Valor Previsto -> [[1.3284395]]
    58 dia. Valores de Entrada -> [1.24806857 1.2464757  1.24788618 1.25197291 1.25847375 1.26705027
     1.27728927 1.28872442 1.30086315 1.31321025 1.32529485 1.33669031
     1.3470335  1.35603726 1.36349249 1.36927199 1.37332666 1.37567461
     1.37639642 1.37561834 1.37350559 1.37025034 1.36606181 1.36115825
     1.3557601  1.35008299 1.34433138 1.33869565 1.33334816 1.32843947]
    58 dia. Valor Previsto -> [[1.3240961]]
    59 dia. Valores de Entrada -> [1.2464757  1.24788618 1.25197291 1.25847375 1.26705027 1.27728927
     1.28872442 1.30086315 1.31321025 1.32529485 1.33669031 1.3470335
     1.35603726 1.36349249 1.36927199 1.37332666 1.37567461 1.37639642
     1.37561834 1.37350559 1.37025034 1.36606181 1.36115825 1.3557601
     1.35008299 1.34433138 1.33869565 1.33334816 1.32843947 1.32409608]
    59 dia. Valor Previsto -> [[1.3204186]]
    60 dia. Valores de Entrada -> [1.24788618 1.25197291 1.25847375 1.26705027 1.27728927 1.28872442
     1.30086315 1.31321025 1.32529485 1.33669031 1.3470335  1.35603726
     1.36349249 1.36927199 1.37332666 1.37567461 1.37639642 1.37561834
     1.37350559 1.37025034 1.36606181 1.36115825 1.3557601  1.35008299
     1.34433138 1.33869565 1.33334816 1.32843947 1.32409608 1.3204186 ]
    60 dia. Valor Previsto -> [[1.3174807]]
    61 dia. Valores de Entrada -> [1.25197291 1.25847375 1.26705027 1.27728927 1.28872442 1.30086315
     1.31321025 1.32529485 1.33669031 1.3470335  1.35603726 1.36349249
     1.36927199 1.37332666 1.37567461 1.37639642 1.37561834 1.37350559
     1.37025034 1.36606181 1.36115825 1.3557601  1.35008299 1.34433138
     1.33869565 1.33334816 1.32843947 1.32409608 1.3204186  1.31748068]
    61 dia. Valor Previsto -> [[1.3153265]]
    62 dia. Valores de Entrada -> [1.25847375 1.26705027 1.27728927 1.28872442 1.30086315 1.31321025
     1.32529485 1.33669031 1.3470335  1.35603726 1.36349249 1.36927199
     1.37332666 1.37567461 1.37639642 1.37561834 1.37350559 1.37025034
     1.36606181 1.36115825 1.3557601  1.35008299 1.34433138 1.33869565
     1.33334816 1.32843947 1.32409608 1.3204186  1.31748068 1.31532645]
    62 dia. Valor Previsto -> [[1.3139735]]
    63 dia. Valores de Entrada -> [1.26705027 1.27728927 1.28872442 1.30086315 1.31321025 1.32529485
     1.33669031 1.3470335  1.35603726 1.36349249 1.36927199 1.37332666
     1.37567461 1.37639642 1.37561834 1.37350559 1.37025034 1.36606181
     1.36115825 1.3557601  1.35008299 1.34433138 1.33869565 1.33334816
     1.32843947 1.32409608 1.3204186  1.31748068 1.31532645 1.31397355]
    63 dia. Valor Previsto -> [[1.3134102]]
    64 dia. Valores de Entrada -> [1.27728927 1.28872442 1.30086315 1.31321025 1.32529485 1.33669031
     1.3470335  1.35603726 1.36349249 1.36927199 1.37332666 1.37567461
     1.37639642 1.37561834 1.37350559 1.37025034 1.36606181 1.36115825
     1.3557601  1.35008299 1.34433138 1.33869565 1.33334816 1.32843947
     1.32409608 1.3204186  1.31748068 1.31532645 1.31397355 1.31341016]
    64 dia. Valor Previsto -> [[1.313602]]
    65 dia. Valores de Entrada -> [1.28872442 1.30086315 1.31321025 1.32529485 1.33669031 1.3470335
     1.35603726 1.36349249 1.36927199 1.37332666 1.37567461 1.37639642
     1.37561834 1.37350559 1.37025034 1.36606181 1.36115825 1.3557601
     1.35008299 1.34433138 1.33869565 1.33334816 1.32843947 1.32409608
     1.3204186  1.31748068 1.31532645 1.31397355 1.31341016 1.31360197]
    65 dia. Valor Previsto -> [[1.3144891]]
    66 dia. Valores de Entrada -> [1.30086315 1.31321025 1.32529485 1.33669031 1.3470335  1.35603726
     1.36349249 1.36927199 1.37332666 1.37567461 1.37639642 1.37561834
     1.37350559 1.37025034 1.36606181 1.36115825 1.3557601  1.35008299
     1.34433138 1.33869565 1.33334816 1.32843947 1.32409608 1.3204186
     1.31748068 1.31532645 1.31397355 1.31341016 1.31360197 1.31448913]
    66 dia. Valor Previsto -> [[1.3159924]]
    67 dia. Valores de Entrada -> [1.31321025 1.32529485 1.33669031 1.3470335  1.35603726 1.36349249
     1.36927199 1.37332666 1.37567461 1.37639642 1.37561834 1.37350559
     1.37025034 1.36606181 1.36115825 1.3557601  1.35008299 1.34433138
     1.33869565 1.33334816 1.32843947 1.32409608 1.3204186  1.31748068
     1.31532645 1.31397355 1.31341016 1.31360197 1.31448913 1.31599236]
    67 dia. Valor Previsto -> [[1.3180162]]
    68 dia. Valores de Entrada -> [1.32529485 1.33669031 1.3470335  1.35603726 1.36349249 1.36927199
     1.37332666 1.37567461 1.37639642 1.37561834 1.37350559 1.37025034
     1.36606181 1.36115825 1.3557601  1.35008299 1.34433138 1.33869565
     1.33334816 1.32843947 1.32409608 1.3204186  1.31748068 1.31532645
     1.31397355 1.31341016 1.31360197 1.31448913 1.31599236 1.31801617]
    68 dia. Valor Previsto -> [[1.320454]]
    69 dia. Valores de Entrada -> [1.33669031 1.3470335  1.35603726 1.36349249 1.36927199 1.37332666
     1.37567461 1.37639642 1.37561834 1.37350559 1.37025034 1.36606181
     1.36115825 1.3557601  1.35008299 1.34433138 1.33869565 1.33334816
     1.32843947 1.32409608 1.3204186  1.31748068 1.31532645 1.31397355
     1.31341016 1.31360197 1.31448913 1.31599236 1.31801617 1.320454  ]
    69 dia. Valor Previsto -> [[1.3231909]]
    70 dia. Valores de Entrada -> [1.3470335  1.35603726 1.36349249 1.36927199 1.37332666 1.37567461
     1.37639642 1.37561834 1.37350559 1.37025034 1.36606181 1.36115825
     1.3557601  1.35008299 1.34433138 1.33869565 1.33334816 1.32843947
     1.32409608 1.3204186  1.31748068 1.31532645 1.31397355 1.31341016
     1.31360197 1.31448913 1.31599236 1.31801617 1.320454   1.32319093]
    70 dia. Valor Previsto -> [[1.3261106]]
    71 dia. Valores de Entrada -> [1.35603726 1.36349249 1.36927199 1.37332666 1.37567461 1.37639642
     1.37561834 1.37350559 1.37025034 1.36606181 1.36115825 1.3557601
     1.35008299 1.34433138 1.33869565 1.33334816 1.32843947 1.32409608
     1.3204186  1.31748068 1.31532645 1.31397355 1.31341016 1.31360197
     1.31448913 1.31599236 1.31801617 1.320454   1.32319093 1.3261106 ]
    71 dia. Valor Previsto -> [[1.3290976]]
    72 dia. Valores de Entrada -> [1.36349249 1.36927199 1.37332666 1.37567461 1.37639642 1.37561834
     1.37350559 1.37025034 1.36606181 1.36115825 1.3557601  1.35008299
     1.34433138 1.33869565 1.33334816 1.32843947 1.32409608 1.3204186
     1.31748068 1.31532645 1.31397355 1.31341016 1.31360197 1.31448913
     1.31599236 1.31801617 1.320454   1.32319093 1.3261106  1.32909763]
    72 dia. Valor Previsto -> [[1.3320436]]
    73 dia. Valores de Entrada -> [1.36927199 1.37332666 1.37567461 1.37639642 1.37561834 1.37350559
     1.37025034 1.36606181 1.36115825 1.3557601  1.35008299 1.34433138
     1.33869565 1.33334816 1.32843947 1.32409608 1.3204186  1.31748068
     1.31532645 1.31397355 1.31341016 1.31360197 1.31448913 1.31599236
     1.31801617 1.320454   1.32319093 1.3261106  1.32909763 1.33204365]
    73 dia. Valor Previsto -> [[1.3348489]]
    74 dia. Valores de Entrada -> [1.37332666 1.37567461 1.37639642 1.37561834 1.37350559 1.37025034
     1.36606181 1.36115825 1.3557601  1.35008299 1.34433138 1.33869565
     1.33334816 1.32843947 1.32409608 1.3204186  1.31748068 1.31532645
     1.31397355 1.31341016 1.31360197 1.31448913 1.31599236 1.31801617
     1.320454   1.32319093 1.3261106  1.32909763 1.33204365 1.33484888]
    74 dia. Valor Previsto -> [[1.3374279]]
    75 dia. Valores de Entrada -> [1.37567461 1.37639642 1.37561834 1.37350559 1.37025034 1.36606181
     1.36115825 1.3557601  1.35008299 1.34433138 1.33869565 1.33334816
     1.32843947 1.32409608 1.3204186  1.31748068 1.31532645 1.31397355
     1.31341016 1.31360197 1.31448913 1.31599236 1.31801617 1.320454
     1.32319093 1.3261106  1.32909763 1.33204365 1.33484888 1.33742785]
    75 dia. Valor Previsto -> [[1.3397089]]
    76 dia. Valores de Entrada -> [1.37639642 1.37561834 1.37350559 1.37025034 1.36606181 1.36115825
     1.3557601  1.35008299 1.34433138 1.33869565 1.33334816 1.32843947
     1.32409608 1.3204186  1.31748068 1.31532645 1.31397355 1.31341016
     1.31360197 1.31448913 1.31599236 1.31801617 1.320454   1.32319093
     1.3261106  1.32909763 1.33204365 1.33484888 1.33742785 1.33970892]
    76 dia. Valor Previsto -> [[1.3416375]]
    77 dia. Valores de Entrada -> [1.37561834 1.37350559 1.37025034 1.36606181 1.36115825 1.3557601
     1.35008299 1.34433138 1.33869565 1.33334816 1.32843947 1.32409608
     1.3204186  1.31748068 1.31532645 1.31397355 1.31341016 1.31360197
     1.31448913 1.31599236 1.31801617 1.320454   1.32319093 1.3261106
     1.32909763 1.33204365 1.33484888 1.33742785 1.33970892 1.34163749]
    77 dia. Valor Previsto -> [[1.3431752]]
    78 dia. Valores de Entrada -> [1.37350559 1.37025034 1.36606181 1.36115825 1.3557601  1.35008299
     1.34433138 1.33869565 1.33334816 1.32843947 1.32409608 1.3204186
     1.31748068 1.31532645 1.31397355 1.31341016 1.31360197 1.31448913
     1.31599236 1.31801617 1.320454   1.32319093 1.3261106  1.32909763
     1.33204365 1.33484888 1.33742785 1.33970892 1.34163749 1.34317517]
    78 dia. Valor Previsto -> [[1.3443017]]
    79 dia. Valores de Entrada -> [1.37025034 1.36606181 1.36115825 1.3557601  1.35008299 1.34433138
     1.33869565 1.33334816 1.32843947 1.32409608 1.3204186  1.31748068
     1.31532645 1.31397355 1.31341016 1.31360197 1.31448913 1.31599236
     1.31801617 1.320454   1.32319093 1.3261106  1.32909763 1.33204365
     1.33484888 1.33742785 1.33970892 1.34163749 1.34317517 1.3443017 ]
    79 dia. Valor Previsto -> [[1.3450106]]
    80 dia. Valores de Entrada -> [1.36606181 1.36115825 1.3557601  1.35008299 1.34433138 1.33869565
     1.33334816 1.32843947 1.32409608 1.3204186  1.31748068 1.31532645
     1.31397355 1.31341016 1.31360197 1.31448913 1.31599236 1.31801617
     1.320454   1.32319093 1.3261106  1.32909763 1.33204365 1.33484888
     1.33742785 1.33970892 1.34163749 1.34317517 1.3443017  1.34501064]
    80 dia. Valor Previsto -> [[1.3453131]]
    81 dia. Valores de Entrada -> [1.36115825 1.3557601  1.35008299 1.34433138 1.33869565 1.33334816
     1.32843947 1.32409608 1.3204186  1.31748068 1.31532645 1.31397355
     1.31341016 1.31360197 1.31448913 1.31599236 1.31801617 1.320454
     1.32319093 1.3261106  1.32909763 1.33204365 1.33484888 1.33742785
     1.33970892 1.34163749 1.34317517 1.3443017  1.34501064 1.34531307]
    81 dia. Valor Previsto -> [[1.3452313]]
    82 dia. Valores de Entrada -> [1.3557601  1.35008299 1.34433138 1.33869565 1.33334816 1.32843947
     1.32409608 1.3204186  1.31748068 1.31532645 1.31397355 1.31341016
     1.31360197 1.31448913 1.31599236 1.31801617 1.320454   1.32319093
     1.3261106  1.32909763 1.33204365 1.33484888 1.33742785 1.33970892
     1.34163749 1.34317517 1.3443017  1.34501064 1.34531307 1.34523129]
    82 dia. Valor Previsto -> [[1.3447989]]
    83 dia. Valores de Entrada -> [1.35008299 1.34433138 1.33869565 1.33334816 1.32843947 1.32409608
     1.3204186  1.31748068 1.31532645 1.31397355 1.31341016 1.31360197
     1.31448913 1.31599236 1.31801617 1.320454   1.32319093 1.3261106
     1.32909763 1.33204365 1.33484888 1.33742785 1.33970892 1.34163749
     1.34317517 1.3443017  1.34501064 1.34531307 1.34523129 1.34479892]
    83 dia. Valor Previsto -> [[1.3440589]]
    84 dia. Valores de Entrada -> [1.34433138 1.33869565 1.33334816 1.32843947 1.32409608 1.3204186
     1.31748068 1.31532645 1.31397355 1.31341016 1.31360197 1.31448913
     1.31599236 1.31801617 1.320454   1.32319093 1.3261106  1.32909763
     1.33204365 1.33484888 1.33742785 1.33970892 1.34163749 1.34317517
     1.3443017  1.34501064 1.34531307 1.34523129 1.34479892 1.34405887]
    84 dia. Valor Previsto -> [[1.3430614]]
    85 dia. Valores de Entrada -> [1.33869565 1.33334816 1.32843947 1.32409608 1.3204186  1.31748068
     1.31532645 1.31397355 1.31341016 1.31360197 1.31448913 1.31599236
     1.31801617 1.320454   1.32319093 1.3261106  1.32909763 1.33204365
     1.33484888 1.33742785 1.33970892 1.34163749 1.34317517 1.3443017
     1.34501064 1.34531307 1.34523129 1.34479892 1.34405887 1.34306145]
    85 dia. Valor Previsto -> [[1.3418595]]
    86 dia. Valores de Entrada -> [1.33334816 1.32843947 1.32409608 1.3204186  1.31748068 1.31532645
     1.31397355 1.31341016 1.31360197 1.31448913 1.31599236 1.31801617
     1.320454   1.32319093 1.3261106  1.32909763 1.33204365 1.33484888
     1.33742785 1.33970892 1.34163749 1.34317517 1.3443017  1.34501064
     1.34531307 1.34523129 1.34479892 1.34405887 1.34306145 1.34185946]
    86 dia. Valor Previsto -> [[1.34051]]
    87 dia. Valores de Entrada -> [1.32843947 1.32409608 1.3204186  1.31748068 1.31532645 1.31397355
     1.31341016 1.31360197 1.31448913 1.31599236 1.31801617 1.320454
     1.32319093 1.3261106  1.32909763 1.33204365 1.33484888 1.33742785
     1.33970892 1.34163749 1.34317517 1.3443017  1.34501064 1.34531307
     1.34523129 1.34479892 1.34405887 1.34306145 1.34185946 1.34051001]
    87 dia. Valor Previsto -> [[1.3390697]]
    88 dia. Valores de Entrada -> [1.32409608 1.3204186  1.31748068 1.31532645 1.31397355 1.31341016
     1.31360197 1.31448913 1.31599236 1.31801617 1.320454   1.32319093
     1.3261106  1.32909763 1.33204365 1.33484888 1.33742785 1.33970892
     1.34163749 1.34317517 1.3443017  1.34501064 1.34531307 1.34523129
     1.34479892 1.34405887 1.34306145 1.34185946 1.34051001 1.33906972]
    88 dia. Valor Previsto -> [[1.3375938]]
    89 dia. Valores de Entrada -> [1.3204186  1.31748068 1.31532645 1.31397355 1.31341016 1.31360197
     1.31448913 1.31599236 1.31801617 1.320454   1.32319093 1.3261106
     1.32909763 1.33204365 1.33484888 1.33742785 1.33970892 1.34163749
     1.34317517 1.3443017  1.34501064 1.34531307 1.34523129 1.34479892
     1.34405887 1.34306145 1.34185946 1.34051001 1.33906972 1.33759379]
    89 dia. Valor Previsto -> [[1.3361337]]
    90 dia. Valores de Entrada -> [1.31748068 1.31532645 1.31397355 1.31341016 1.31360197 1.31448913
     1.31599236 1.31801617 1.320454   1.32319093 1.3261106  1.32909763
     1.33204365 1.33484888 1.33742785 1.33970892 1.34163749 1.34317517
     1.3443017  1.34501064 1.34531307 1.34523129 1.34479892 1.34405887
     1.34306145 1.34185946 1.34051001 1.33906972 1.33759379 1.33613372]
    90 dia. Valor Previsto -> [[1.3347378]]
    91 dia. Valores de Entrada -> [1.31532645 1.31397355 1.31341016 1.31360197 1.31448913 1.31599236
     1.31801617 1.320454   1.32319093 1.3261106  1.32909763 1.33204365
     1.33484888 1.33742785 1.33970892 1.34163749 1.34317517 1.3443017
     1.34501064 1.34531307 1.34523129 1.34479892 1.34405887 1.34306145
     1.34185946 1.34051001 1.33906972 1.33759379 1.33613372 1.33473778]
    91 dia. Valor Previsto -> [[1.3334472]]
    92 dia. Valores de Entrada -> [1.31397355 1.31341016 1.31360197 1.31448913 1.31599236 1.31801617
     1.320454   1.32319093 1.3261106  1.32909763 1.33204365 1.33484888
     1.33742785 1.33970892 1.34163749 1.34317517 1.3443017  1.34501064
     1.34531307 1.34523129 1.34479892 1.34405887 1.34306145 1.34185946
     1.34051001 1.33906972 1.33759379 1.33613372 1.33473778 1.33344722]
    92 dia. Valor Previsto -> [[1.3322971]]
    93 dia. Valores de Entrada -> [1.31341016 1.31360197 1.31448913 1.31599236 1.31801617 1.320454
     1.32319093 1.3261106  1.32909763 1.33204365 1.33484888 1.33742785
     1.33970892 1.34163749 1.34317517 1.3443017  1.34501064 1.34531307
     1.34523129 1.34479892 1.34405887 1.34306145 1.34185946 1.34051001
     1.33906972 1.33759379 1.33613372 1.33473778 1.33344722 1.33229709]
    93 dia. Valor Previsto -> [[1.3313154]]
    94 dia. Valores de Entrada -> [1.31360197 1.31448913 1.31599236 1.31801617 1.320454   1.32319093
     1.3261106  1.32909763 1.33204365 1.33484888 1.33742785 1.33970892
     1.34163749 1.34317517 1.3443017  1.34501064 1.34531307 1.34523129
     1.34479892 1.34405887 1.34306145 1.34185946 1.34051001 1.33906972
     1.33759379 1.33613372 1.33473778 1.33344722 1.33229709 1.3313154 ]
    94 dia. Valor Previsto -> [[1.3305215]]
    95 dia. Valores de Entrada -> [1.31448913 1.31599236 1.31801617 1.320454   1.32319093 1.3261106
     1.32909763 1.33204365 1.33484888 1.33742785 1.33970892 1.34163749
     1.34317517 1.3443017  1.34501064 1.34531307 1.34523129 1.34479892
     1.34405887 1.34306145 1.34185946 1.34051001 1.33906972 1.33759379
     1.33613372 1.33473778 1.33344722 1.33229709 1.3313154  1.33052146]
    95 dia. Valor Previsto -> [[1.3299292]]
    96 dia. Valores de Entrada -> [1.31599236 1.31801617 1.320454   1.32319093 1.3261106  1.32909763
     1.33204365 1.33484888 1.33742785 1.33970892 1.34163749 1.34317517
     1.3443017  1.34501064 1.34531307 1.34523129 1.34479892 1.34405887
     1.34306145 1.34185946 1.34051001 1.33906972 1.33759379 1.33613372
     1.33473778 1.33344722 1.33229709 1.3313154  1.33052146 1.32992923]
    96 dia. Valor Previsto -> [[1.329542]]
    97 dia. Valores de Entrada -> [1.31801617 1.320454   1.32319093 1.3261106  1.32909763 1.33204365
     1.33484888 1.33742785 1.33970892 1.34163749 1.34317517 1.3443017
     1.34501064 1.34531307 1.34523129 1.34479892 1.34405887 1.34306145
     1.34185946 1.34051001 1.33906972 1.33759379 1.33613372 1.33473778
     1.33344722 1.33229709 1.3313154  1.33052146 1.32992923 1.32954204]
    97 dia. Valor Previsto -> [[1.3293582]]
    98 dia. Valores de Entrada -> [1.320454   1.32319093 1.3261106  1.32909763 1.33204365 1.33484888
     1.33742785 1.33970892 1.34163749 1.34317517 1.3443017  1.34501064
     1.34531307 1.34523129 1.34479892 1.34405887 1.34306145 1.34185946
     1.34051001 1.33906972 1.33759379 1.33613372 1.33473778 1.33344722
     1.33229709 1.3313154  1.33052146 1.32992923 1.32954204 1.32935822]
    98 dia. Valor Previsto -> [[1.329369]]
    99 dia. Valores de Entrada -> [1.32319093 1.3261106  1.32909763 1.33204365 1.33484888 1.33742785
     1.33970892 1.34163749 1.34317517 1.3443017  1.34501064 1.34531307
     1.34523129 1.34479892 1.34405887 1.34306145 1.34185946 1.34051001
     1.33906972 1.33759379 1.33613372 1.33473778 1.33344722 1.33229709
     1.3313154  1.33052146 1.32992923 1.32954204 1.32935822 1.32936895]
    99 dia. Valor Previsto -> [[1.3295584]]
    100 dia. Valores de Entrada -> [1.3261106  1.32909763 1.33204365 1.33484888 1.33742785 1.33970892
     1.34163749 1.34317517 1.3443017  1.34501064 1.34531307 1.34523129
     1.34479892 1.34405887 1.34306145 1.34185946 1.34051001 1.33906972
     1.33759379 1.33613372 1.33473778 1.33344722 1.33229709 1.3313154
     1.33052146 1.32992923 1.32954204 1.32935822 1.32936895 1.32955837]
    100 dia. Valor Previsto -> [[1.3299071]]
    101 dia. Valores de Entrada -> [1.32909763 1.33204365 1.33484888 1.33742785 1.33970892 1.34163749
     1.34317517 1.3443017  1.34501064 1.34531307 1.34523129 1.34479892
     1.34405887 1.34306145 1.34185946 1.34051001 1.33906972 1.33759379
     1.33613372 1.33473778 1.33344722 1.33229709 1.3313154  1.33052146
     1.32992923 1.32954204 1.32935822 1.32936895 1.32955837 1.32990706]
    101 dia. Valor Previsto -> [[1.330391]]
    102 dia. Valores de Entrada -> [1.33204365 1.33484888 1.33742785 1.33970892 1.34163749 1.34317517
     1.3443017  1.34501064 1.34531307 1.34523129 1.34479892 1.34405887
     1.34306145 1.34185946 1.34051001 1.33906972 1.33759379 1.33613372
     1.33473778 1.33344722 1.33229709 1.3313154  1.33052146 1.32992923
     1.32954204 1.32935822 1.32936895 1.32955837 1.32990706 1.33039105]
    102 dia. Valor Previsto -> [[1.3309839]]
    103 dia. Valores de Entrada -> [1.33484888 1.33742785 1.33970892 1.34163749 1.34317517 1.3443017
     1.34501064 1.34531307 1.34523129 1.34479892 1.34405887 1.34306145
     1.34185946 1.34051001 1.33906972 1.33759379 1.33613372 1.33473778
     1.33344722 1.33229709 1.3313154  1.33052146 1.32992923 1.32954204
     1.32935822 1.32936895 1.32955837 1.32990706 1.33039105 1.33098388]
    103 dia. Valor Previsto -> [[1.3316569]]
    104 dia. Valores de Entrada -> [1.33742785 1.33970892 1.34163749 1.34317517 1.3443017  1.34501064
     1.34531307 1.34523129 1.34479892 1.34405887 1.34306145 1.34185946
     1.34051001 1.33906972 1.33759379 1.33613372 1.33473778 1.33344722
     1.33229709 1.3313154  1.33052146 1.32992923 1.32954204 1.32935822
     1.32936895 1.32955837 1.32990706 1.33039105 1.33098388 1.33165693]
    104 dia. Valor Previsto -> [[1.3323805]]
    105 dia. Valores de Entrada -> [1.33970892 1.34163749 1.34317517 1.3443017  1.34501064 1.34531307
     1.34523129 1.34479892 1.34405887 1.34306145 1.34185946 1.34051001
     1.33906972 1.33759379 1.33613372 1.33473778 1.33344722 1.33229709
     1.3313154  1.33052146 1.32992923 1.32954204 1.32935822 1.32936895
     1.32955837 1.32990706 1.33039105 1.33098388 1.33165693 1.33238053]
    105 dia. Valor Previsto -> [[1.3331274]]
    106 dia. Valores de Entrada -> [1.34163749 1.34317517 1.3443017  1.34501064 1.34531307 1.34523129
     1.34479892 1.34405887 1.34306145 1.34185946 1.34051001 1.33906972
     1.33759379 1.33613372 1.33473778 1.33344722 1.33229709 1.3313154
     1.33052146 1.32992923 1.32954204 1.32935822 1.32936895 1.32955837
     1.32990706 1.33039105 1.33098388 1.33165693 1.33238053 1.33312738]
    106 dia. Valor Previsto -> [[1.3338692]]
    107 dia. Valores de Entrada -> [1.34317517 1.3443017  1.34501064 1.34531307 1.34523129 1.34479892
     1.34405887 1.34306145 1.34185946 1.34051001 1.33906972 1.33759379
     1.33613372 1.33473778 1.33344722 1.33229709 1.3313154  1.33052146
     1.32992923 1.32954204 1.32935822 1.32936895 1.32955837 1.32990706
     1.33039105 1.33098388 1.33165693 1.33238053 1.33312738 1.33386922]
    107 dia. Valor Previsto -> [[1.3345822]]
    108 dia. Valores de Entrada -> [1.3443017  1.34501064 1.34531307 1.34523129 1.34479892 1.34405887
     1.34306145 1.34185946 1.34051001 1.33906972 1.33759379 1.33613372
     1.33473778 1.33344722 1.33229709 1.3313154  1.33052146 1.32992923
     1.32954204 1.32935822 1.32936895 1.32955837 1.32990706 1.33039105
     1.33098388 1.33165693 1.33238053 1.33312738 1.33386922 1.33458221]
    108 dia. Valor Previsto -> [[1.3352439]]
    109 dia. Valores de Entrada -> [1.34501064 1.34531307 1.34523129 1.34479892 1.34405887 1.34306145
     1.34185946 1.34051001 1.33906972 1.33759379 1.33613372 1.33473778
     1.33344722 1.33229709 1.3313154  1.33052146 1.32992923 1.32954204
     1.32935822 1.32936895 1.32955837 1.32990706 1.33039105 1.33098388
     1.33165693 1.33238053 1.33312738 1.33386922 1.33458221 1.33524394]
    109 dia. Valor Previsto -> [[1.3358358]]
    110 dia. Valores de Entrada -> [1.34531307 1.34523129 1.34479892 1.34405887 1.34306145 1.34185946
     1.34051001 1.33906972 1.33759379 1.33613372 1.33473778 1.33344722
     1.33229709 1.3313154  1.33052146 1.32992923 1.32954204 1.32935822
     1.32936895 1.32955837 1.32990706 1.33039105 1.33098388 1.33165693
     1.33238053 1.33312738 1.33386922 1.33458221 1.33524394 1.33583581]
    110 dia. Valor Previsto -> [[1.3363428]]
    111 dia. Valores de Entrada -> [1.34523129 1.34479892 1.34405887 1.34306145 1.34185946 1.34051001
     1.33906972 1.33759379 1.33613372 1.33473778 1.33344722 1.33229709
     1.3313154  1.33052146 1.32992923 1.32954204 1.32935822 1.32936895
     1.32955837 1.32990706 1.33039105 1.33098388 1.33165693 1.33238053
     1.33312738 1.33386922 1.33458221 1.33524394 1.33583581 1.33634281]
    111 dia. Valor Previsto -> [[1.3367546]]
    112 dia. Valores de Entrada -> [1.34479892 1.34405887 1.34306145 1.34185946 1.34051001 1.33906972
     1.33759379 1.33613372 1.33473778 1.33344722 1.33229709 1.3313154
     1.33052146 1.32992923 1.32954204 1.32935822 1.32936895 1.32955837
     1.32990706 1.33039105 1.33098388 1.33165693 1.33238053 1.33312738
     1.33386922 1.33458221 1.33524394 1.33583581 1.33634281 1.33675456]
    112 dia. Valor Previsto -> [[1.337065]]
    113 dia. Valores de Entrada -> [1.34405887 1.34306145 1.34185946 1.34051001 1.33906972 1.33759379
     1.33613372 1.33473778 1.33344722 1.33229709 1.3313154  1.33052146
     1.32992923 1.32954204 1.32935822 1.32936895 1.32955837 1.32990706
     1.33039105 1.33098388 1.33165693 1.33238053 1.33312738 1.33386922
     1.33458221 1.33524394 1.33583581 1.33634281 1.33675456 1.33706498]
    113 dia. Valor Previsto -> [[1.3372712]]
    114 dia. Valores de Entrada -> [1.34306145 1.34185946 1.34051001 1.33906972 1.33759379 1.33613372
     1.33473778 1.33344722 1.33229709 1.3313154  1.33052146 1.32992923
     1.32954204 1.32935822 1.32936895 1.32955837 1.32990706 1.33039105
     1.33098388 1.33165693 1.33238053 1.33312738 1.33386922 1.33458221
     1.33524394 1.33583581 1.33634281 1.33675456 1.33706498 1.33727121]
    114 dia. Valor Previsto -> [[1.337374]]
    115 dia. Valores de Entrada -> [1.34185946 1.34051001 1.33906972 1.33759379 1.33613372 1.33473778
     1.33344722 1.33229709 1.3313154  1.33052146 1.32992923 1.32954204
     1.32935822 1.32936895 1.32955837 1.32990706 1.33039105 1.33098388
     1.33165693 1.33238053 1.33312738 1.33386922 1.33458221 1.33524394
     1.33583581 1.33634281 1.33675456 1.33706498 1.33727121 1.33737397]
    115 dia. Valor Previsto -> [[1.337378]]
    116 dia. Valores de Entrada -> [1.34051001 1.33906972 1.33759379 1.33613372 1.33473778 1.33344722
     1.33229709 1.3313154  1.33052146 1.32992923 1.32954204 1.32935822
     1.32936895 1.32955837 1.32990706 1.33039105 1.33098388 1.33165693
     1.33238053 1.33312738 1.33386922 1.33458221 1.33524394 1.33583581
     1.33634281 1.33675456 1.33706498 1.33727121 1.33737397 1.33737803]
    116 dia. Valor Previsto -> [[1.3372917]]
    117 dia. Valores de Entrada -> [1.33906972 1.33759379 1.33613372 1.33473778 1.33344722 1.33229709
     1.3313154  1.33052146 1.32992923 1.32954204 1.32935822 1.32936895
     1.32955837 1.32990706 1.33039105 1.33098388 1.33165693 1.33238053
     1.33312738 1.33386922 1.33458221 1.33524394 1.33583581 1.33634281
     1.33675456 1.33706498 1.33727121 1.33737397 1.33737803 1.33729172]
    117 dia. Valor Previsto -> [[1.3371243]]
    118 dia. Valores de Entrada -> [1.33759379 1.33613372 1.33473778 1.33344722 1.33229709 1.3313154
     1.33052146 1.32992923 1.32954204 1.32935822 1.32936895 1.32955837
     1.32990706 1.33039105 1.33098388 1.33165693 1.33238053 1.33312738
     1.33386922 1.33458221 1.33524394 1.33583581 1.33634281 1.33675456
     1.33706498 1.33727121 1.33737397 1.33737803 1.33729172 1.33712435]
    118 dia. Valor Previsto -> [[1.3368881]]
    119 dia. Valores de Entrada -> [1.33613372 1.33473778 1.33344722 1.33229709 1.3313154  1.33052146
     1.32992923 1.32954204 1.32935822 1.32936895 1.32955837 1.32990706
     1.33039105 1.33098388 1.33165693 1.33238053 1.33312738 1.33386922
     1.33458221 1.33524394 1.33583581 1.33634281 1.33675456 1.33706498
     1.33727121 1.33737397 1.33737803 1.33729172 1.33712435 1.33688807]
    119 dia. Valor Previsto -> [[1.3365961]]
    120 dia. Valores de Entrada -> [1.33473778 1.33344722 1.33229709 1.3313154  1.33052146 1.32992923
     1.32954204 1.32935822 1.32936895 1.32955837 1.32990706 1.33039105
     1.33098388 1.33165693 1.33238053 1.33312738 1.33386922 1.33458221
     1.33524394 1.33583581 1.33634281 1.33675456 1.33706498 1.33727121
     1.33737397 1.33737803 1.33729172 1.33712435 1.33688807 1.33659613]
    120 dia. Valor Previsto -> [[1.3362628]]
    121 dia. Valores de Entrada -> [1.33344722 1.33229709 1.3313154  1.33052146 1.32992923 1.32954204
     1.32935822 1.32936895 1.32955837 1.32990706 1.33039105 1.33098388
     1.33165693 1.33238053 1.33312738 1.33386922 1.33458221 1.33524394
     1.33583581 1.33634281 1.33675456 1.33706498 1.33727121 1.33737397
     1.33737803 1.33729172 1.33712435 1.33688807 1.33659613 1.33626282]
    121 dia. Valor Previsto -> [[1.3359025]]
    122 dia. Valores de Entrada -> [1.33229709 1.3313154  1.33052146 1.32992923 1.32954204 1.32935822
     1.32936895 1.32955837 1.32990706 1.33039105 1.33098388 1.33165693
     1.33238053 1.33312738 1.33386922 1.33458221 1.33524394 1.33583581
     1.33634281 1.33675456 1.33706498 1.33727121 1.33737397 1.33737803
     1.33729172 1.33712435 1.33688807 1.33659613 1.33626282 1.33590245]
    122 dia. Valor Previsto -> [[1.3355293]]
    123 dia. Valores de Entrada -> [1.3313154  1.33052146 1.32992923 1.32954204 1.32935822 1.32936895
     1.32955837 1.32990706 1.33039105 1.33098388 1.33165693 1.33238053
     1.33312738 1.33386922 1.33458221 1.33524394 1.33583581 1.33634281
     1.33675456 1.33706498 1.33727121 1.33737397 1.33737803 1.33729172
     1.33712435 1.33688807 1.33659613 1.33626282 1.33590245 1.33552933]
    123 dia. Valor Previsto -> [[1.3351569]]
    124 dia. Valores de Entrada -> [1.33052146 1.32992923 1.32954204 1.32935822 1.32936895 1.32955837
     1.32990706 1.33039105 1.33098388 1.33165693 1.33238053 1.33312738
     1.33386922 1.33458221 1.33524394 1.33583581 1.33634281 1.33675456
     1.33706498 1.33727121 1.33737397 1.33737803 1.33729172 1.33712435
     1.33688807 1.33659613 1.33626282 1.33590245 1.33552933 1.33515692]
    124 dia. Valor Previsto -> [[1.3347975]]
    125 dia. Valores de Entrada -> [1.32992923 1.32954204 1.32935822 1.32936895 1.32955837 1.32990706
     1.33039105 1.33098388 1.33165693 1.33238053 1.33312738 1.33386922
     1.33458221 1.33524394 1.33583581 1.33634281 1.33675456 1.33706498
     1.33727121 1.33737397 1.33737803 1.33729172 1.33712435 1.33688807
     1.33659613 1.33626282 1.33590245 1.33552933 1.33515692 1.3347975 ]
    125 dia. Valor Previsto -> [[1.3344625]]
    126 dia. Valores de Entrada -> [1.32954204 1.32935822 1.32936895 1.32955837 1.32990706 1.33039105
     1.33098388 1.33165693 1.33238053 1.33312738 1.33386922 1.33458221
     1.33524394 1.33583581 1.33634281 1.33675456 1.33706498 1.33727121
     1.33737397 1.33737803 1.33729172 1.33712435 1.33688807 1.33659613
     1.33626282 1.33590245 1.33552933 1.33515692 1.3347975  1.33446252]
    126 dia. Valor Previsto -> [[1.3341619]]
    127 dia. Valores de Entrada -> [1.32935822 1.32936895 1.32955837 1.32990706 1.33039105 1.33098388
     1.33165693 1.33238053 1.33312738 1.33386922 1.33458221 1.33524394
     1.33583581 1.33634281 1.33675456 1.33706498 1.33727121 1.33737397
     1.33737803 1.33729172 1.33712435 1.33688807 1.33659613 1.33626282
     1.33590245 1.33552933 1.33515692 1.3347975  1.33446252 1.33416188]
    127 dia. Valor Previsto -> [[1.3339022]]
    128 dia. Valores de Entrada -> [1.32936895 1.32955837 1.32990706 1.33039105 1.33098388 1.33165693
     1.33238053 1.33312738 1.33386922 1.33458221 1.33524394 1.33583581
     1.33634281 1.33675456 1.33706498 1.33727121 1.33737397 1.33737803
     1.33729172 1.33712435 1.33688807 1.33659613 1.33626282 1.33590245
     1.33552933 1.33515692 1.3347975  1.33446252 1.33416188 1.33390224]
    128 dia. Valor Previsto -> [[1.3336896]]
    129 dia. Valores de Entrada -> [1.32955837 1.32990706 1.33039105 1.33098388 1.33165693 1.33238053
     1.33312738 1.33386922 1.33458221 1.33524394 1.33583581 1.33634281
     1.33675456 1.33706498 1.33727121 1.33737397 1.33737803 1.33729172
     1.33712435 1.33688807 1.33659613 1.33626282 1.33590245 1.33552933
     1.33515692 1.3347975  1.33446252 1.33416188 1.33390224 1.33368957]
    129 dia. Valor Previsto -> [[1.333527]]
    130 dia. Valores de Entrada -> [1.32990706 1.33039105 1.33098388 1.33165693 1.33238053 1.33312738
     1.33386922 1.33458221 1.33524394 1.33583581 1.33634281 1.33675456
     1.33706498 1.33727121 1.33737397 1.33737803 1.33729172 1.33712435
     1.33688807 1.33659613 1.33626282 1.33590245 1.33552933 1.33515692
     1.3347975  1.33446252 1.33416188 1.33390224 1.33368957 1.33352697]
    130 dia. Valor Previsto -> [[1.3334172]]
    131 dia. Valores de Entrada -> [1.33039105 1.33098388 1.33165693 1.33238053 1.33312738 1.33386922
     1.33458221 1.33524394 1.33583581 1.33634281 1.33675456 1.33706498
     1.33727121 1.33737397 1.33737803 1.33729172 1.33712435 1.33688807
     1.33659613 1.33626282 1.33590245 1.33552933 1.33515692 1.3347975
     1.33446252 1.33416188 1.33390224 1.33368957 1.33352697 1.33341718]
    131 dia. Valor Previsto -> [[1.3333584]]
    132 dia. Valores de Entrada -> [1.33098388 1.33165693 1.33238053 1.33312738 1.33386922 1.33458221
     1.33524394 1.33583581 1.33634281 1.33675456 1.33706498 1.33727121
     1.33737397 1.33737803 1.33729172 1.33712435 1.33688807 1.33659613
     1.33626282 1.33590245 1.33552933 1.33515692 1.3347975  1.33446252
     1.33416188 1.33390224 1.33368957 1.33352697 1.33341718 1.33335841]
    132 dia. Valor Previsto -> [[1.3333502]]
    133 dia. Valores de Entrada -> [1.33165693 1.33238053 1.33312738 1.33386922 1.33458221 1.33524394
     1.33583581 1.33634281 1.33675456 1.33706498 1.33727121 1.33737397
     1.33737803 1.33729172 1.33712435 1.33688807 1.33659613 1.33626282
     1.33590245 1.33552933 1.33515692 1.3347975  1.33446252 1.33416188
     1.33390224 1.33368957 1.33352697 1.33341718 1.33335841 1.33335018]
    133 dia. Valor Previsto -> [[1.3333877]]
    134 dia. Valores de Entrada -> [1.33238053 1.33312738 1.33386922 1.33458221 1.33524394 1.33583581
     1.33634281 1.33675456 1.33706498 1.33727121 1.33737397 1.33737803
     1.33729172 1.33712435 1.33688807 1.33659613 1.33626282 1.33590245
     1.33552933 1.33515692 1.3347975  1.33446252 1.33416188 1.33390224
     1.33368957 1.33352697 1.33341718 1.33335841 1.33335018 1.33338773]
    134 dia. Valor Previsto -> [[1.3334666]]
    135 dia. Valores de Entrada -> [1.33312738 1.33386922 1.33458221 1.33524394 1.33583581 1.33634281
     1.33675456 1.33706498 1.33727121 1.33737397 1.33737803 1.33729172
     1.33712435 1.33688807 1.33659613 1.33626282 1.33590245 1.33552933
     1.33515692 1.3347975  1.33446252 1.33416188 1.33390224 1.33368957
     1.33352697 1.33341718 1.33335841 1.33335018 1.33338773 1.33346665]
    135 dia. Valor Previsto -> [[1.3335813]]
    136 dia. Valores de Entrada -> [1.33386922 1.33458221 1.33524394 1.33583581 1.33634281 1.33675456
     1.33706498 1.33727121 1.33737397 1.33737803 1.33729172 1.33712435
     1.33688807 1.33659613 1.33626282 1.33590245 1.33552933 1.33515692
     1.3347975  1.33446252 1.33416188 1.33390224 1.33368957 1.33352697
     1.33341718 1.33335841 1.33335018 1.33338773 1.33346665 1.33358133]
    136 dia. Valor Previsto -> [[1.3337244]]
    137 dia. Valores de Entrada -> [1.33458221 1.33524394 1.33583581 1.33634281 1.33675456 1.33706498
     1.33727121 1.33737397 1.33737803 1.33729172 1.33712435 1.33688807
     1.33659613 1.33626282 1.33590245 1.33552933 1.33515692 1.3347975
     1.33446252 1.33416188 1.33390224 1.33368957 1.33352697 1.33341718
     1.33335841 1.33335018 1.33338773 1.33346665 1.33358133 1.33372438]
    137 dia. Valor Previsto -> [[1.3338898]]
    138 dia. Valores de Entrada -> [1.33524394 1.33583581 1.33634281 1.33675456 1.33706498 1.33727121
     1.33737397 1.33737803 1.33729172 1.33712435 1.33688807 1.33659613
     1.33626282 1.33590245 1.33552933 1.33515692 1.3347975  1.33446252
     1.33416188 1.33390224 1.33368957 1.33352697 1.33341718 1.33335841
     1.33335018 1.33338773 1.33346665 1.33358133 1.33372438 1.33388984]
    138 dia. Valor Previsto -> [[1.33407]]
    139 dia. Valores de Entrada -> [1.33583581 1.33634281 1.33675456 1.33706498 1.33727121 1.33737397
     1.33737803 1.33729172 1.33712435 1.33688807 1.33659613 1.33626282
     1.33590245 1.33552933 1.33515692 1.3347975  1.33446252 1.33416188
     1.33390224 1.33368957 1.33352697 1.33341718 1.33335841 1.33335018
     1.33338773 1.33346665 1.33358133 1.33372438 1.33388984 1.33406997]
    139 dia. Valor Previsto -> [[1.3342574]]
    140 dia. Valores de Entrada -> [1.33634281 1.33675456 1.33706498 1.33727121 1.33737397 1.33737803
     1.33729172 1.33712435 1.33688807 1.33659613 1.33626282 1.33590245
     1.33552933 1.33515692 1.3347975  1.33446252 1.33416188 1.33390224
     1.33368957 1.33352697 1.33341718 1.33335841 1.33335018 1.33338773
     1.33346665 1.33358133 1.33372438 1.33388984 1.33406997 1.33425736]
    140 dia. Valor Previsto -> [[1.3344452]]
    141 dia. Valores de Entrada -> [1.33675456 1.33706498 1.33727121 1.33737397 1.33737803 1.33729172
     1.33712435 1.33688807 1.33659613 1.33626282 1.33590245 1.33552933
     1.33515692 1.3347975  1.33446252 1.33416188 1.33390224 1.33368957
     1.33352697 1.33341718 1.33335841 1.33335018 1.33338773 1.33346665
     1.33358133 1.33372438 1.33388984 1.33406997 1.33425736 1.33444524]
    141 dia. Valor Previsto -> [[1.3346274]]
    142 dia. Valores de Entrada -> [1.33706498 1.33727121 1.33737397 1.33737803 1.33729172 1.33712435
     1.33688807 1.33659613 1.33626282 1.33590245 1.33552933 1.33515692
     1.3347975  1.33446252 1.33416188 1.33390224 1.33368957 1.33352697
     1.33341718 1.33335841 1.33335018 1.33338773 1.33346665 1.33358133
     1.33372438 1.33388984 1.33406997 1.33425736 1.33444524 1.33462739]
    142 dia. Valor Previsto -> [[1.3347976]]
    143 dia. Valores de Entrada -> [1.33727121 1.33737397 1.33737803 1.33729172 1.33712435 1.33688807
     1.33659613 1.33626282 1.33590245 1.33552933 1.33515692 1.3347975
     1.33446252 1.33416188 1.33390224 1.33368957 1.33352697 1.33341718
     1.33335841 1.33335018 1.33338773 1.33346665 1.33358133 1.33372438
     1.33388984 1.33406997 1.33425736 1.33444524 1.33462739 1.33479762]
    143 dia. Valor Previsto -> [[1.3349514]]
    144 dia. Valores de Entrada -> [1.33737397 1.33737803 1.33729172 1.33712435 1.33688807 1.33659613
     1.33626282 1.33590245 1.33552933 1.33515692 1.3347975  1.33446252
     1.33416188 1.33390224 1.33368957 1.33352697 1.33341718 1.33335841
     1.33335018 1.33338773 1.33346665 1.33358133 1.33372438 1.33388984
     1.33406997 1.33425736 1.33444524 1.33462739 1.33479762 1.3349514 ]
    144 dia. Valor Previsto -> [[1.3350846]]
    145 dia. Valores de Entrada -> [1.33737803 1.33729172 1.33712435 1.33688807 1.33659613 1.33626282
     1.33590245 1.33552933 1.33515692 1.3347975  1.33446252 1.33416188
     1.33390224 1.33368957 1.33352697 1.33341718 1.33335841 1.33335018
     1.33338773 1.33346665 1.33358133 1.33372438 1.33388984 1.33406997
     1.33425736 1.33444524 1.33462739 1.33479762 1.3349514  1.33508456]
    145 dia. Valor Previsto -> [[1.3351948]]
    146 dia. Valores de Entrada -> [1.33729172 1.33712435 1.33688807 1.33659613 1.33626282 1.33590245
     1.33552933 1.33515692 1.3347975  1.33446252 1.33416188 1.33390224
     1.33368957 1.33352697 1.33341718 1.33335841 1.33335018 1.33338773
     1.33346665 1.33358133 1.33372438 1.33388984 1.33406997 1.33425736
     1.33444524 1.33462739 1.33479762 1.3349514  1.33508456 1.33519483]
    146 dia. Valor Previsto -> [[1.3352795]]
    147 dia. Valores de Entrada -> [1.33712435 1.33688807 1.33659613 1.33626282 1.33590245 1.33552933
     1.33515692 1.3347975  1.33446252 1.33416188 1.33390224 1.33368957
     1.33352697 1.33341718 1.33335841 1.33335018 1.33338773 1.33346665
     1.33358133 1.33372438 1.33388984 1.33406997 1.33425736 1.33444524
     1.33462739 1.33479762 1.3349514  1.33508456 1.33519483 1.33527946]
    147 dia. Valor Previsto -> [[1.3353384]]
    148 dia. Valores de Entrada -> [1.33688807 1.33659613 1.33626282 1.33590245 1.33552933 1.33515692
     1.3347975  1.33446252 1.33416188 1.33390224 1.33368957 1.33352697
     1.33341718 1.33335841 1.33335018 1.33338773 1.33346665 1.33358133
     1.33372438 1.33388984 1.33406997 1.33425736 1.33444524 1.33462739
     1.33479762 1.3349514  1.33508456 1.33519483 1.33527946 1.33533835]
    148 dia. Valor Previsto -> [[1.335371]]
    149 dia. Valores de Entrada -> [1.33659613 1.33626282 1.33590245 1.33552933 1.33515692 1.3347975
     1.33446252 1.33416188 1.33390224 1.33368957 1.33352697 1.33341718
     1.33335841 1.33335018 1.33338773 1.33346665 1.33358133 1.33372438
     1.33388984 1.33406997 1.33425736 1.33444524 1.33462739 1.33479762
     1.3349514  1.33508456 1.33519483 1.33527946 1.33533835 1.33537102]
    149 dia. Valor Previsto -> [[1.3353785]]
    150 dia. Valores de Entrada -> [1.33626282 1.33590245 1.33552933 1.33515692 1.3347975  1.33446252
     1.33416188 1.33390224 1.33368957 1.33352697 1.33341718 1.33335841
     1.33335018 1.33338773 1.33346665 1.33358133 1.33372438 1.33388984
     1.33406997 1.33425736 1.33444524 1.33462739 1.33479762 1.3349514
     1.33508456 1.33519483 1.33527946 1.33533835 1.33537102 1.33537853]
    150 dia. Valor Previsto -> [[1.3353622]]
    151 dia. Valores de Entrada -> [1.33590245 1.33552933 1.33515692 1.3347975  1.33446252 1.33416188
     1.33390224 1.33368957 1.33352697 1.33341718 1.33335841 1.33335018
     1.33338773 1.33346665 1.33358133 1.33372438 1.33388984 1.33406997
     1.33425736 1.33444524 1.33462739 1.33479762 1.3349514  1.33508456
     1.33519483 1.33527946 1.33533835 1.33537102 1.33537853 1.3353622 ]
    151 dia. Valor Previsto -> [[1.335325]]
    152 dia. Valores de Entrada -> [1.33552933 1.33515692 1.3347975  1.33446252 1.33416188 1.33390224
     1.33368957 1.33352697 1.33341718 1.33335841 1.33335018 1.33338773
     1.33346665 1.33358133 1.33372438 1.33388984 1.33406997 1.33425736
     1.33444524 1.33462739 1.33479762 1.3349514  1.33508456 1.33519483
     1.33527946 1.33533835 1.33537102 1.33537853 1.3353622  1.335325  ]
    152 dia. Valor Previsto -> [[1.3352692]]
    153 dia. Valores de Entrada -> [1.33515692 1.3347975  1.33446252 1.33416188 1.33390224 1.33368957
     1.33352697 1.33341718 1.33335841 1.33335018 1.33338773 1.33346665
     1.33358133 1.33372438 1.33388984 1.33406997 1.33425736 1.33444524
     1.33462739 1.33479762 1.3349514  1.33508456 1.33519483 1.33527946
     1.33533835 1.33537102 1.33537853 1.3353622  1.335325   1.33526921]
    153 dia. Valor Previsto -> [[1.3351988]]
    154 dia. Valores de Entrada -> [1.3347975  1.33446252 1.33416188 1.33390224 1.33368957 1.33352697
     1.33341718 1.33335841 1.33335018 1.33338773 1.33346665 1.33358133
     1.33372438 1.33388984 1.33406997 1.33425736 1.33444524 1.33462739
     1.33479762 1.3349514  1.33508456 1.33519483 1.33527946 1.33533835
     1.33537102 1.33537853 1.3353622  1.335325   1.33526921 1.33519876]
    154 dia. Valor Previsto -> [[1.3351167]]
    155 dia. Valores de Entrada -> [1.33446252 1.33416188 1.33390224 1.33368957 1.33352697 1.33341718
     1.33335841 1.33335018 1.33338773 1.33346665 1.33358133 1.33372438
     1.33388984 1.33406997 1.33425736 1.33444524 1.33462739 1.33479762
     1.3349514  1.33508456 1.33519483 1.33527946 1.33533835 1.33537102
     1.33537853 1.3353622  1.335325   1.33526921 1.33519876 1.33511674]
    155 dia. Valor Previsto -> [[1.335027]]
    156 dia. Valores de Entrada -> [1.33416188 1.33390224 1.33368957 1.33352697 1.33341718 1.33335841
     1.33335018 1.33338773 1.33346665 1.33358133 1.33372438 1.33388984
     1.33406997 1.33425736 1.33444524 1.33462739 1.33479762 1.3349514
     1.33508456 1.33519483 1.33527946 1.33533835 1.33537102 1.33537853
     1.3353622  1.335325   1.33526921 1.33519876 1.33511674 1.33502698]
    156 dia. Valor Previsto -> [[1.3349334]]
    157 dia. Valores de Entrada -> [1.33390224 1.33368957 1.33352697 1.33341718 1.33335841 1.33335018
     1.33338773 1.33346665 1.33358133 1.33372438 1.33388984 1.33406997
     1.33425736 1.33444524 1.33462739 1.33479762 1.3349514  1.33508456
     1.33519483 1.33527946 1.33533835 1.33537102 1.33537853 1.3353622
     1.335325   1.33526921 1.33519876 1.33511674 1.33502698 1.3349334 ]
    157 dia. Valor Previsto -> [[1.3348387]]
    158 dia. Valores de Entrada -> [1.33368957 1.33352697 1.33341718 1.33335841 1.33335018 1.33338773
     1.33346665 1.33358133 1.33372438 1.33388984 1.33406997 1.33425736
     1.33444524 1.33462739 1.33479762 1.3349514  1.33508456 1.33519483
     1.33527946 1.33533835 1.33537102 1.33537853 1.3353622  1.335325
     1.33526921 1.33519876 1.33511674 1.33502698 1.3349334  1.33483875]
    158 dia. Valor Previsto -> [[1.334747]]
    159 dia. Valores de Entrada -> [1.33352697 1.33341718 1.33335841 1.33335018 1.33338773 1.33346665
     1.33358133 1.33372438 1.33388984 1.33406997 1.33425736 1.33444524
     1.33462739 1.33479762 1.3349514  1.33508456 1.33519483 1.33527946
     1.33533835 1.33537102 1.33537853 1.3353622  1.335325   1.33526921
     1.33519876 1.33511674 1.33502698 1.3349334  1.33483875 1.33474696]
    159 dia. Valor Previsto -> [[1.3346606]]
    160 dia. Valores de Entrada -> [1.33341718 1.33335841 1.33335018 1.33338773 1.33346665 1.33358133
     1.33372438 1.33388984 1.33406997 1.33425736 1.33444524 1.33462739
     1.33479762 1.3349514  1.33508456 1.33519483 1.33527946 1.33533835
     1.33537102 1.33537853 1.3353622  1.335325   1.33526921 1.33519876
     1.33511674 1.33502698 1.3349334  1.33483875 1.33474696 1.33466065]
    160 dia. Valor Previsto -> [[1.334582]]
    161 dia. Valores de Entrada -> [1.33335841 1.33335018 1.33338773 1.33346665 1.33358133 1.33372438
     1.33388984 1.33406997 1.33425736 1.33444524 1.33462739 1.33479762
     1.3349514  1.33508456 1.33519483 1.33527946 1.33533835 1.33537102
     1.33537853 1.3353622  1.335325   1.33526921 1.33519876 1.33511674
     1.33502698 1.3349334  1.33483875 1.33474696 1.33466065 1.33458197]
    161 dia. Valor Previsto -> [[1.3345137]]
    162 dia. Valores de Entrada -> [1.33335018 1.33338773 1.33346665 1.33358133 1.33372438 1.33388984
     1.33406997 1.33425736 1.33444524 1.33462739 1.33479762 1.3349514
     1.33508456 1.33519483 1.33527946 1.33533835 1.33537102 1.33537853
     1.3353622  1.335325   1.33526921 1.33519876 1.33511674 1.33502698
     1.3349334  1.33483875 1.33474696 1.33466065 1.33458197 1.33451366]
    162 dia. Valor Previsto -> [[1.3344568]]
    163 dia. Valores de Entrada -> [1.33338773 1.33346665 1.33358133 1.33372438 1.33388984 1.33406997
     1.33425736 1.33444524 1.33462739 1.33479762 1.3349514  1.33508456
     1.33519483 1.33527946 1.33533835 1.33537102 1.33537853 1.3353622
     1.335325   1.33526921 1.33519876 1.33511674 1.33502698 1.3349334
     1.33483875 1.33474696 1.33466065 1.33458197 1.33451366 1.3344568 ]
    163 dia. Valor Previsto -> [[1.3344127]]
    164 dia. Valores de Entrada -> [1.33346665 1.33358133 1.33372438 1.33388984 1.33406997 1.33425736
     1.33444524 1.33462739 1.33479762 1.3349514  1.33508456 1.33519483
     1.33527946 1.33533835 1.33537102 1.33537853 1.3353622  1.335325
     1.33526921 1.33519876 1.33511674 1.33502698 1.3349334  1.33483875
     1.33474696 1.33466065 1.33458197 1.33451366 1.3344568  1.33441269]
    164 dia. Valor Previsto -> [[1.3343813]]
    165 dia. Valores de Entrada -> [1.33358133 1.33372438 1.33388984 1.33406997 1.33425736 1.33444524
     1.33462739 1.33479762 1.3349514  1.33508456 1.33519483 1.33527946
     1.33533835 1.33537102 1.33537853 1.3353622  1.335325   1.33526921
     1.33519876 1.33511674 1.33502698 1.3349334  1.33483875 1.33474696
     1.33466065 1.33458197 1.33451366 1.3344568  1.33441269 1.33438134]
    165 dia. Valor Previsto -> [[1.3343635]]
    166 dia. Valores de Entrada -> [1.33372438 1.33388984 1.33406997 1.33425736 1.33444524 1.33462739
     1.33479762 1.3349514  1.33508456 1.33519483 1.33527946 1.33533835
     1.33537102 1.33537853 1.3353622  1.335325   1.33526921 1.33519876
     1.33511674 1.33502698 1.3349334  1.33483875 1.33474696 1.33466065
     1.33458197 1.33451366 1.3344568  1.33441269 1.33438134 1.33436346]
    166 dia. Valor Previsto -> [[1.3343577]]
    167 dia. Valores de Entrada -> [1.33388984 1.33406997 1.33425736 1.33444524 1.33462739 1.33479762
     1.3349514  1.33508456 1.33519483 1.33527946 1.33533835 1.33537102
     1.33537853 1.3353622  1.335325   1.33526921 1.33519876 1.33511674
     1.33502698 1.3349334  1.33483875 1.33474696 1.33466065 1.33458197
     1.33451366 1.3344568  1.33441269 1.33438134 1.33436346 1.33435774]
    167 dia. Valor Previsto -> [[1.3343648]]
    168 dia. Valores de Entrada -> [1.33406997 1.33425736 1.33444524 1.33462739 1.33479762 1.3349514
     1.33508456 1.33519483 1.33527946 1.33533835 1.33537102 1.33537853
     1.3353622  1.335325   1.33526921 1.33519876 1.33511674 1.33502698
     1.3349334  1.33483875 1.33474696 1.33466065 1.33458197 1.33451366
     1.3344568  1.33441269 1.33438134 1.33436346 1.33435774 1.33436477]
    168 dia. Valor Previsto -> [[1.3343823]]
    169 dia. Valores de Entrada -> [1.33425736 1.33444524 1.33462739 1.33479762 1.3349514  1.33508456
     1.33519483 1.33527946 1.33533835 1.33537102 1.33537853 1.3353622
     1.335325   1.33526921 1.33519876 1.33511674 1.33502698 1.3349334
     1.33483875 1.33474696 1.33466065 1.33458197 1.33451366 1.3344568
     1.33441269 1.33438134 1.33436346 1.33435774 1.33436477 1.3343823 ]
    169 dia. Valor Previsto -> [[1.3344089]]
    170 dia. Valores de Entrada -> [1.33444524 1.33462739 1.33479762 1.3349514  1.33508456 1.33519483
     1.33527946 1.33533835 1.33537102 1.33537853 1.3353622  1.335325
     1.33526921 1.33519876 1.33511674 1.33502698 1.3349334  1.33483875
     1.33474696 1.33466065 1.33458197 1.33451366 1.3344568  1.33441269
     1.33438134 1.33436346 1.33435774 1.33436477 1.3343823  1.33440888]
    170 dia. Valor Previsto -> [[1.3344434]]
    171 dia. Valores de Entrada -> [1.33462739 1.33479762 1.3349514  1.33508456 1.33519483 1.33527946
     1.33533835 1.33537102 1.33537853 1.3353622  1.335325   1.33526921
     1.33519876 1.33511674 1.33502698 1.3349334  1.33483875 1.33474696
     1.33466065 1.33458197 1.33451366 1.3344568  1.33441269 1.33438134
     1.33436346 1.33435774 1.33436477 1.3343823  1.33440888 1.33444345]
    171 dia. Valor Previsto -> [[1.3344842]]
    172 dia. Valores de Entrada -> [1.33479762 1.3349514  1.33508456 1.33519483 1.33527946 1.33533835
     1.33537102 1.33537853 1.3353622  1.335325   1.33526921 1.33519876
     1.33511674 1.33502698 1.3349334  1.33483875 1.33474696 1.33466065
     1.33458197 1.33451366 1.3344568  1.33441269 1.33438134 1.33436346
     1.33435774 1.33436477 1.3343823  1.33440888 1.33444345 1.33448422]
    172 dia. Valor Previsto -> [[1.3345292]]
    173 dia. Valores de Entrada -> [1.3349514  1.33508456 1.33519483 1.33527946 1.33533835 1.33537102
     1.33537853 1.3353622  1.335325   1.33526921 1.33519876 1.33511674
     1.33502698 1.3349334  1.33483875 1.33474696 1.33466065 1.33458197
     1.33451366 1.3344568  1.33441269 1.33438134 1.33436346 1.33435774
     1.33436477 1.3343823  1.33440888 1.33444345 1.33448422 1.33452916]
    173 dia. Valor Previsto -> [[1.3345761]]
    174 dia. Valores de Entrada -> [1.33508456 1.33519483 1.33527946 1.33533835 1.33537102 1.33537853
     1.3353622  1.335325   1.33526921 1.33519876 1.33511674 1.33502698
     1.3349334  1.33483875 1.33474696 1.33466065 1.33458197 1.33451366
     1.3344568  1.33441269 1.33438134 1.33436346 1.33435774 1.33436477
     1.3343823  1.33440888 1.33444345 1.33448422 1.33452916 1.33457613]
    174 dia. Valor Previsto -> [[1.3346237]]
    175 dia. Valores de Entrada -> [1.33519483 1.33527946 1.33533835 1.33537102 1.33537853 1.3353622
     1.335325   1.33526921 1.33519876 1.33511674 1.33502698 1.3349334
     1.33483875 1.33474696 1.33466065 1.33458197 1.33451366 1.3344568
     1.33441269 1.33438134 1.33436346 1.33435774 1.33436477 1.3343823
     1.33440888 1.33444345 1.33448422 1.33452916 1.33457613 1.33462369]
    175 dia. Valor Previsto -> [[1.3346701]]
    176 dia. Valores de Entrada -> [1.33527946 1.33533835 1.33537102 1.33537853 1.3353622  1.335325
     1.33526921 1.33519876 1.33511674 1.33502698 1.3349334  1.33483875
     1.33474696 1.33466065 1.33458197 1.33451366 1.3344568  1.33441269
     1.33438134 1.33436346 1.33435774 1.33436477 1.3343823  1.33440888
     1.33444345 1.33448422 1.33452916 1.33457613 1.33462369 1.33467007]
    176 dia. Valor Previsto -> [[1.3347139]]
    177 dia. Valores de Entrada -> [1.33533835 1.33537102 1.33537853 1.3353622  1.335325   1.33526921
     1.33519876 1.33511674 1.33502698 1.3349334  1.33483875 1.33474696
     1.33466065 1.33458197 1.33451366 1.3344568  1.33441269 1.33438134
     1.33436346 1.33435774 1.33436477 1.3343823  1.33440888 1.33444345
     1.33448422 1.33452916 1.33457613 1.33462369 1.33467007 1.33471394]
    177 dia. Valor Previsto -> [[1.334754]]
    178 dia. Valores de Entrada -> [1.33537102 1.33537853 1.3353622  1.335325   1.33526921 1.33519876
     1.33511674 1.33502698 1.3349334  1.33483875 1.33474696 1.33466065
     1.33458197 1.33451366 1.3344568  1.33441269 1.33438134 1.33436346
     1.33435774 1.33436477 1.3343823  1.33440888 1.33444345 1.33448422
     1.33452916 1.33457613 1.33462369 1.33467007 1.33471394 1.33475399]
    178 dia. Valor Previsto -> [[1.334789]]
    179 dia. Valores de Entrada -> [1.33537853 1.3353622  1.335325   1.33526921 1.33519876 1.33511674
     1.33502698 1.3349334  1.33483875 1.33474696 1.33466065 1.33458197
     1.33451366 1.3344568  1.33441269 1.33438134 1.33436346 1.33435774
     1.33436477 1.3343823  1.33440888 1.33444345 1.33448422 1.33452916
     1.33457613 1.33462369 1.33467007 1.33471394 1.33475399 1.33478904]
    179 dia. Valor Previsto -> [[1.3348182]]
    180 dia. Valores de Entrada -> [1.3353622  1.335325   1.33526921 1.33519876 1.33511674 1.33502698
     1.3349334  1.33483875 1.33474696 1.33466065 1.33458197 1.33451366
     1.3344568  1.33441269 1.33438134 1.33436346 1.33435774 1.33436477
     1.3343823  1.33440888 1.33444345 1.33448422 1.33452916 1.33457613
     1.33462369 1.33467007 1.33471394 1.33475399 1.33478904 1.33481824]
    180 dia. Valor Previsto -> [[1.3348416]]
    181 dia. Valores de Entrada -> [1.335325   1.33526921 1.33519876 1.33511674 1.33502698 1.3349334
     1.33483875 1.33474696 1.33466065 1.33458197 1.33451366 1.3344568
     1.33441269 1.33438134 1.33436346 1.33435774 1.33436477 1.3343823
     1.33440888 1.33444345 1.33448422 1.33452916 1.33457613 1.33462369
     1.33467007 1.33471394 1.33475399 1.33478904 1.33481824 1.33484161]
    181 dia. Valor Previsto -> [[1.3348576]]
    182 dia. Valores de Entrada -> [1.33526921 1.33519876 1.33511674 1.33502698 1.3349334  1.33483875
     1.33474696 1.33466065 1.33458197 1.33451366 1.3344568  1.33441269
     1.33438134 1.33436346 1.33435774 1.33436477 1.3343823  1.33440888
     1.33444345 1.33448422 1.33452916 1.33457613 1.33462369 1.33467007
     1.33471394 1.33475399 1.33478904 1.33481824 1.33484161 1.33485758]
    182 dia. Valor Previsto -> [[1.3348676]]
    183 dia. Valores de Entrada -> [1.33519876 1.33511674 1.33502698 1.3349334  1.33483875 1.33474696
     1.33466065 1.33458197 1.33451366 1.3344568  1.33441269 1.33438134
     1.33436346 1.33435774 1.33436477 1.3343823  1.33440888 1.33444345
     1.33448422 1.33452916 1.33457613 1.33462369 1.33467007 1.33471394
     1.33475399 1.33478904 1.33481824 1.33484161 1.33485758 1.3348676 ]
    183 dia. Valor Previsto -> [[1.3348712]]
    184 dia. Valores de Entrada -> [1.33511674 1.33502698 1.3349334  1.33483875 1.33474696 1.33466065
     1.33458197 1.33451366 1.3344568  1.33441269 1.33438134 1.33436346
     1.33435774 1.33436477 1.3343823  1.33440888 1.33444345 1.33448422
     1.33452916 1.33457613 1.33462369 1.33467007 1.33471394 1.33475399
     1.33478904 1.33481824 1.33484161 1.33485758 1.3348676  1.33487117]
    184 dia. Valor Previsto -> [[1.3348686]]
    185 dia. Valores de Entrada -> [1.33502698 1.3349334  1.33483875 1.33474696 1.33466065 1.33458197
     1.33451366 1.3344568  1.33441269 1.33438134 1.33436346 1.33435774
     1.33436477 1.3343823  1.33440888 1.33444345 1.33448422 1.33452916
     1.33457613 1.33462369 1.33467007 1.33471394 1.33475399 1.33478904
     1.33481824 1.33484161 1.33485758 1.3348676  1.33487117 1.33486855]
    185 dia. Valor Previsto -> [[1.3348606]]
    186 dia. Valores de Entrada -> [1.3349334  1.33483875 1.33474696 1.33466065 1.33458197 1.33451366
     1.3344568  1.33441269 1.33438134 1.33436346 1.33435774 1.33436477
     1.3343823  1.33440888 1.33444345 1.33448422 1.33452916 1.33457613
     1.33462369 1.33467007 1.33471394 1.33475399 1.33478904 1.33481824
     1.33484161 1.33485758 1.3348676  1.33487117 1.33486855 1.33486056]
    186 dia. Valor Previsto -> [[1.3348475]]
    187 dia. Valores de Entrada -> [1.33483875 1.33474696 1.33466065 1.33458197 1.33451366 1.3344568
     1.33441269 1.33438134 1.33436346 1.33435774 1.33436477 1.3343823
     1.33440888 1.33444345 1.33448422 1.33452916 1.33457613 1.33462369
     1.33467007 1.33471394 1.33475399 1.33478904 1.33481824 1.33484161
     1.33485758 1.3348676  1.33487117 1.33486855 1.33486056 1.33484745]
    187 dia. Valor Previsto -> [[1.3348306]]
    188 dia. Valores de Entrada -> [1.33474696 1.33466065 1.33458197 1.33451366 1.3344568  1.33441269
     1.33438134 1.33436346 1.33435774 1.33436477 1.3343823  1.33440888
     1.33444345 1.33448422 1.33452916 1.33457613 1.33462369 1.33467007
     1.33471394 1.33475399 1.33478904 1.33481824 1.33484161 1.33485758
     1.3348676  1.33487117 1.33486855 1.33486056 1.33484745 1.33483064]
    188 dia. Valor Previsto -> [[1.3348104]]
    189 dia. Valores de Entrada -> [1.33466065 1.33458197 1.33451366 1.3344568  1.33441269 1.33438134
     1.33436346 1.33435774 1.33436477 1.3343823  1.33440888 1.33444345
     1.33448422 1.33452916 1.33457613 1.33462369 1.33467007 1.33471394
     1.33475399 1.33478904 1.33481824 1.33484161 1.33485758 1.3348676
     1.33487117 1.33486855 1.33486056 1.33484745 1.33483064 1.33481038]
    189 dia. Valor Previsto -> [[1.3347881]]
    190 dia. Valores de Entrada -> [1.33458197 1.33451366 1.3344568  1.33441269 1.33438134 1.33436346
     1.33435774 1.33436477 1.3343823  1.33440888 1.33444345 1.33448422
     1.33452916 1.33457613 1.33462369 1.33467007 1.33471394 1.33475399
     1.33478904 1.33481824 1.33484161 1.33485758 1.3348676  1.33487117
     1.33486855 1.33486056 1.33484745 1.33483064 1.33481038 1.33478808]
    190 dia. Valor Previsto -> [[1.3347647]]
    191 dia. Valores de Entrada -> [1.33451366 1.3344568  1.33441269 1.33438134 1.33436346 1.33435774
     1.33436477 1.3343823  1.33440888 1.33444345 1.33448422 1.33452916
     1.33457613 1.33462369 1.33467007 1.33471394 1.33475399 1.33478904
     1.33481824 1.33484161 1.33485758 1.3348676  1.33487117 1.33486855
     1.33486056 1.33484745 1.33483064 1.33481038 1.33478808 1.33476472]
    191 dia. Valor Previsto -> [[1.3347404]]
    192 dia. Valores de Entrada -> [1.3344568  1.33441269 1.33438134 1.33436346 1.33435774 1.33436477
     1.3343823  1.33440888 1.33444345 1.33448422 1.33452916 1.33457613
     1.33462369 1.33467007 1.33471394 1.33475399 1.33478904 1.33481824
     1.33484161 1.33485758 1.3348676  1.33487117 1.33486855 1.33486056
     1.33484745 1.33483064 1.33481038 1.33478808 1.33476472 1.3347404 ]
    192 dia. Valor Previsto -> [[1.3347172]]
    193 dia. Valores de Entrada -> [1.33441269 1.33438134 1.33436346 1.33435774 1.33436477 1.3343823
     1.33440888 1.33444345 1.33448422 1.33452916 1.33457613 1.33462369
     1.33467007 1.33471394 1.33475399 1.33478904 1.33481824 1.33484161
     1.33485758 1.3348676  1.33487117 1.33486855 1.33486056 1.33484745
     1.33483064 1.33481038 1.33478808 1.33476472 1.3347404  1.33471715]
    193 dia. Valor Previsto -> [[1.3346951]]
    194 dia. Valores de Entrada -> [1.33438134 1.33436346 1.33435774 1.33436477 1.3343823  1.33440888
     1.33444345 1.33448422 1.33452916 1.33457613 1.33462369 1.33467007
     1.33471394 1.33475399 1.33478904 1.33481824 1.33484161 1.33485758
     1.3348676  1.33487117 1.33486855 1.33486056 1.33484745 1.33483064
     1.33481038 1.33478808 1.33476472 1.3347404  1.33471715 1.3346951 ]
    194 dia. Valor Previsto -> [[1.3346748]]
    195 dia. Valores de Entrada -> [1.33436346 1.33435774 1.33436477 1.3343823  1.33440888 1.33444345
     1.33448422 1.33452916 1.33457613 1.33462369 1.33467007 1.33471394
     1.33475399 1.33478904 1.33481824 1.33484161 1.33485758 1.3348676
     1.33487117 1.33486855 1.33486056 1.33484745 1.33483064 1.33481038
     1.33478808 1.33476472 1.3347404  1.33471715 1.3346951  1.33467484]
    195 dia. Valor Previsto -> [[1.334657]]
    196 dia. Valores de Entrada -> [1.33435774 1.33436477 1.3343823  1.33440888 1.33444345 1.33448422
     1.33452916 1.33457613 1.33462369 1.33467007 1.33471394 1.33475399
     1.33478904 1.33481824 1.33484161 1.33485758 1.3348676  1.33487117
     1.33486855 1.33486056 1.33484745 1.33483064 1.33481038 1.33478808
     1.33476472 1.3347404  1.33471715 1.3346951  1.33467484 1.33465695]
    196 dia. Valor Previsto -> [[1.3346416]]
    197 dia. Valores de Entrada -> [1.33436477 1.3343823  1.33440888 1.33444345 1.33448422 1.33452916
     1.33457613 1.33462369 1.33467007 1.33471394 1.33475399 1.33478904
     1.33481824 1.33484161 1.33485758 1.3348676  1.33487117 1.33486855
     1.33486056 1.33484745 1.33483064 1.33481038 1.33478808 1.33476472
     1.3347404  1.33471715 1.3346951  1.33467484 1.33465695 1.33464158]
    197 dia. Valor Previsto -> [[1.3346298]]
    198 dia. Valores de Entrada -> [1.3343823  1.33440888 1.33444345 1.33448422 1.33452916 1.33457613
     1.33462369 1.33467007 1.33471394 1.33475399 1.33478904 1.33481824
     1.33484161 1.33485758 1.3348676  1.33487117 1.33486855 1.33486056
     1.33484745 1.33483064 1.33481038 1.33478808 1.33476472 1.3347404
     1.33471715 1.3346951  1.33467484 1.33465695 1.33464158 1.33462977]
    198 dia. Valor Previsto -> [[1.334621]]
    199 dia. Valores de Entrada -> [1.33440888 1.33444345 1.33448422 1.33452916 1.33457613 1.33462369
     1.33467007 1.33471394 1.33475399 1.33478904 1.33481824 1.33484161
     1.33485758 1.3348676  1.33487117 1.33486855 1.33486056 1.33484745
     1.33483064 1.33481038 1.33478808 1.33476472 1.3347404  1.33471715
     1.3346951  1.33467484 1.33465695 1.33464158 1.33462977 1.33462095]
    199 dia. Valor Previsto -> [[1.3346158]]
    200 dia. Valores de Entrada -> [1.33444345 1.33448422 1.33452916 1.33457613 1.33462369 1.33467007
     1.33471394 1.33475399 1.33478904 1.33481824 1.33484161 1.33485758
     1.3348676  1.33487117 1.33486855 1.33486056 1.33484745 1.33483064
     1.33481038 1.33478808 1.33476472 1.3347404  1.33471715 1.3346951
     1.33467484 1.33465695 1.33464158 1.33462977 1.33462095 1.33461583]
    200 dia. Valor Previsto -> [[1.3346137]]
    201 dia. Valores de Entrada -> [1.33448422 1.33452916 1.33457613 1.33462369 1.33467007 1.33471394
     1.33475399 1.33478904 1.33481824 1.33484161 1.33485758 1.3348676
     1.33487117 1.33486855 1.33486056 1.33484745 1.33483064 1.33481038
     1.33478808 1.33476472 1.3347404  1.33471715 1.3346951  1.33467484
     1.33465695 1.33464158 1.33462977 1.33462095 1.33461583 1.33461368]
    201 dia. Valor Previsto -> [[1.3346149]]
    202 dia. Valores de Entrada -> [1.33452916 1.33457613 1.33462369 1.33467007 1.33471394 1.33475399
     1.33478904 1.33481824 1.33484161 1.33485758 1.3348676  1.33487117
     1.33486855 1.33486056 1.33484745 1.33483064 1.33481038 1.33478808
     1.33476472 1.3347404  1.33471715 1.3346951  1.33467484 1.33465695
     1.33464158 1.33462977 1.33462095 1.33461583 1.33461368 1.33461487]
    202 dia. Valor Previsto -> [[1.3346184]]
    203 dia. Valores de Entrada -> [1.33457613 1.33462369 1.33467007 1.33471394 1.33475399 1.33478904
     1.33481824 1.33484161 1.33485758 1.3348676  1.33487117 1.33486855
     1.33486056 1.33484745 1.33483064 1.33481038 1.33478808 1.33476472
     1.3347404  1.33471715 1.3346951  1.33467484 1.33465695 1.33464158
     1.33462977 1.33462095 1.33461583 1.33461368 1.33461487 1.33461845]
    203 dia. Valor Previsto -> [[1.3346246]]
    204 dia. Valores de Entrada -> [1.33462369 1.33467007 1.33471394 1.33475399 1.33478904 1.33481824
     1.33484161 1.33485758 1.3348676  1.33487117 1.33486855 1.33486056
     1.33484745 1.33483064 1.33481038 1.33478808 1.33476472 1.3347404
     1.33471715 1.3346951  1.33467484 1.33465695 1.33464158 1.33462977
     1.33462095 1.33461583 1.33461368 1.33461487 1.33461845 1.33462465]
    204 dia. Valor Previsto -> [[1.3346329]]
    205 dia. Valores de Entrada -> [1.33467007 1.33471394 1.33475399 1.33478904 1.33481824 1.33484161
     1.33485758 1.3348676  1.33487117 1.33486855 1.33486056 1.33484745
     1.33483064 1.33481038 1.33478808 1.33476472 1.3347404  1.33471715
     1.3346951  1.33467484 1.33465695 1.33464158 1.33462977 1.33462095
     1.33461583 1.33461368 1.33461487 1.33461845 1.33462465 1.33463287]
    205 dia. Valor Previsto -> [[1.3346426]]
    206 dia. Valores de Entrada -> [1.33471394 1.33475399 1.33478904 1.33481824 1.33484161 1.33485758
     1.3348676  1.33487117 1.33486855 1.33486056 1.33484745 1.33483064
     1.33481038 1.33478808 1.33476472 1.3347404  1.33471715 1.3346951
     1.33467484 1.33465695 1.33464158 1.33462977 1.33462095 1.33461583
     1.33461368 1.33461487 1.33461845 1.33462465 1.33463287 1.33464265]
    206 dia. Valor Previsto -> [[1.3346535]]
    207 dia. Valores de Entrada -> [1.33475399 1.33478904 1.33481824 1.33484161 1.33485758 1.3348676
     1.33487117 1.33486855 1.33486056 1.33484745 1.33483064 1.33481038
     1.33478808 1.33476472 1.3347404  1.33471715 1.3346951  1.33467484
     1.33465695 1.33464158 1.33462977 1.33462095 1.33461583 1.33461368
     1.33461487 1.33461845 1.33462465 1.33463287 1.33464265 1.3346535 ]
    207 dia. Valor Previsto -> [[1.3346654]]
    208 dia. Valores de Entrada -> [1.33478904 1.33481824 1.33484161 1.33485758 1.3348676  1.33487117
     1.33486855 1.33486056 1.33484745 1.33483064 1.33481038 1.33478808
     1.33476472 1.3347404  1.33471715 1.3346951  1.33467484 1.33465695
     1.33464158 1.33462977 1.33462095 1.33461583 1.33461368 1.33461487
     1.33461845 1.33462465 1.33463287 1.33464265 1.3346535  1.33466542]
    208 dia. Valor Previsto -> [[1.3346776]]
    209 dia. Valores de Entrada -> [1.33481824 1.33484161 1.33485758 1.3348676  1.33487117 1.33486855
     1.33486056 1.33484745 1.33483064 1.33481038 1.33478808 1.33476472
     1.3347404  1.33471715 1.3346951  1.33467484 1.33465695 1.33464158
     1.33462977 1.33462095 1.33461583 1.33461368 1.33461487 1.33461845
     1.33462465 1.33463287 1.33464265 1.3346535  1.33466542 1.33467758]
    209 dia. Valor Previsto -> [[1.3346894]]
    210 dia. Valores de Entrada -> [1.33484161 1.33485758 1.3348676  1.33487117 1.33486855 1.33486056
     1.33484745 1.33483064 1.33481038 1.33478808 1.33476472 1.3347404
     1.33471715 1.3346951  1.33467484 1.33465695 1.33464158 1.33462977
     1.33462095 1.33461583 1.33461368 1.33461487 1.33461845 1.33462465
     1.33463287 1.33464265 1.3346535  1.33466542 1.33467758 1.33468938]
    210 dia. Valor Previsto -> [[1.3347007]]
    211 dia. Valores de Entrada -> [1.33485758 1.3348676  1.33487117 1.33486855 1.33486056 1.33484745
     1.33483064 1.33481038 1.33478808 1.33476472 1.3347404  1.33471715
     1.3346951  1.33467484 1.33465695 1.33464158 1.33462977 1.33462095
     1.33461583 1.33461368 1.33461487 1.33461845 1.33462465 1.33463287
     1.33464265 1.3346535  1.33466542 1.33467758 1.33468938 1.3347007 ]
    211 dia. Valor Previsto -> [[1.3347112]]
    212 dia. Valores de Entrada -> [1.3348676  1.33487117 1.33486855 1.33486056 1.33484745 1.33483064
     1.33481038 1.33478808 1.33476472 1.3347404  1.33471715 1.3346951
     1.33467484 1.33465695 1.33464158 1.33462977 1.33462095 1.33461583
     1.33461368 1.33461487 1.33461845 1.33462465 1.33463287 1.33464265
     1.3346535  1.33466542 1.33467758 1.33468938 1.3347007  1.33471119]
    212 dia. Valor Previsto -> [[1.3347205]]
    213 dia. Valores de Entrada -> [1.33487117 1.33486855 1.33486056 1.33484745 1.33483064 1.33481038
     1.33478808 1.33476472 1.3347404  1.33471715 1.3346951  1.33467484
     1.33465695 1.33464158 1.33462977 1.33462095 1.33461583 1.33461368
     1.33461487 1.33461845 1.33462465 1.33463287 1.33464265 1.3346535
     1.33466542 1.33467758 1.33468938 1.3347007  1.33471119 1.33472049]
    213 dia. Valor Previsto -> [[1.334728]]
    214 dia. Valores de Entrada -> [1.33486855 1.33486056 1.33484745 1.33483064 1.33481038 1.33478808
     1.33476472 1.3347404  1.33471715 1.3346951  1.33467484 1.33465695
     1.33464158 1.33462977 1.33462095 1.33461583 1.33461368 1.33461487
     1.33461845 1.33462465 1.33463287 1.33464265 1.3346535  1.33466542
     1.33467758 1.33468938 1.3347007  1.33471119 1.33472049 1.334728  ]
    214 dia. Valor Previsto -> [[1.3347342]]
    215 dia. Valores de Entrada -> [1.33486056 1.33484745 1.33483064 1.33481038 1.33478808 1.33476472
     1.3347404  1.33471715 1.3346951  1.33467484 1.33465695 1.33464158
     1.33462977 1.33462095 1.33461583 1.33461368 1.33461487 1.33461845
     1.33462465 1.33463287 1.33464265 1.3346535  1.33466542 1.33467758
     1.33468938 1.3347007  1.33471119 1.33472049 1.334728   1.3347342 ]
    215 dia. Valor Previsto -> [[1.3347389]]
    216 dia. Valores de Entrada -> [1.33484745 1.33483064 1.33481038 1.33478808 1.33476472 1.3347404
     1.33471715 1.3346951  1.33467484 1.33465695 1.33464158 1.33462977
     1.33462095 1.33461583 1.33461368 1.33461487 1.33461845 1.33462465
     1.33463287 1.33464265 1.3346535  1.33466542 1.33467758 1.33468938
     1.3347007  1.33471119 1.33472049 1.334728   1.3347342  1.33473885]
    216 dia. Valor Previsto -> [[1.3347415]]
    217 dia. Valores de Entrada -> [1.33483064 1.33481038 1.33478808 1.33476472 1.3347404  1.33471715
     1.3346951  1.33467484 1.33465695 1.33464158 1.33462977 1.33462095
     1.33461583 1.33461368 1.33461487 1.33461845 1.33462465 1.33463287
     1.33464265 1.3346535  1.33466542 1.33467758 1.33468938 1.3347007
     1.33471119 1.33472049 1.334728   1.3347342  1.33473885 1.33474147]
    217 dia. Valor Previsto -> [[1.3347425]]
    218 dia. Valores de Entrada -> [1.33481038 1.33478808 1.33476472 1.3347404  1.33471715 1.3346951
     1.33467484 1.33465695 1.33464158 1.33462977 1.33462095 1.33461583
     1.33461368 1.33461487 1.33461845 1.33462465 1.33463287 1.33464265
     1.3346535  1.33466542 1.33467758 1.33468938 1.3347007  1.33471119
     1.33472049 1.334728   1.3347342  1.33473885 1.33474147 1.33474255]
    218 dia. Valor Previsto -> [[1.3347425]]
    219 dia. Valores de Entrada -> [1.33478808 1.33476472 1.3347404  1.33471715 1.3346951  1.33467484
     1.33465695 1.33464158 1.33462977 1.33462095 1.33461583 1.33461368
     1.33461487 1.33461845 1.33462465 1.33463287 1.33464265 1.3346535
     1.33466542 1.33467758 1.33468938 1.3347007  1.33471119 1.33472049
     1.334728   1.3347342  1.33473885 1.33474147 1.33474255 1.33474255]
    219 dia. Valor Previsto -> [[1.3347408]]
    220 dia. Valores de Entrada -> [1.33476472 1.3347404  1.33471715 1.3346951  1.33467484 1.33465695
     1.33464158 1.33462977 1.33462095 1.33461583 1.33461368 1.33461487
     1.33461845 1.33462465 1.33463287 1.33464265 1.3346535  1.33466542
     1.33467758 1.33468938 1.3347007  1.33471119 1.33472049 1.334728
     1.3347342  1.33473885 1.33474147 1.33474255 1.33474255 1.33474076]
    220 dia. Valor Previsto -> [[1.3347378]]
    221 dia. Valores de Entrada -> [1.3347404  1.33471715 1.3346951  1.33467484 1.33465695 1.33464158
     1.33462977 1.33462095 1.33461583 1.33461368 1.33461487 1.33461845
     1.33462465 1.33463287 1.33464265 1.3346535  1.33466542 1.33467758
     1.33468938 1.3347007  1.33471119 1.33472049 1.334728   1.3347342
     1.33473885 1.33474147 1.33474255 1.33474255 1.33474076 1.33473778]
    221 dia. Valor Previsto -> [[1.3347336]]
    222 dia. Valores de Entrada -> [1.33471715 1.3346951  1.33467484 1.33465695 1.33464158 1.33462977
     1.33462095 1.33461583 1.33461368 1.33461487 1.33461845 1.33462465
     1.33463287 1.33464265 1.3346535  1.33466542 1.33467758 1.33468938
     1.3347007  1.33471119 1.33472049 1.334728   1.3347342  1.33473885
     1.33474147 1.33474255 1.33474255 1.33474076 1.33473778 1.33473361]
    222 dia. Valor Previsto -> [[1.3347292]]
    223 dia. Valores de Entrada -> [1.3346951  1.33467484 1.33465695 1.33464158 1.33462977 1.33462095
     1.33461583 1.33461368 1.33461487 1.33461845 1.33462465 1.33463287
     1.33464265 1.3346535  1.33466542 1.33467758 1.33468938 1.3347007
     1.33471119 1.33472049 1.334728   1.3347342  1.33473885 1.33474147
     1.33474255 1.33474255 1.33474076 1.33473778 1.33473361 1.33472919]
    223 dia. Valor Previsto -> [[1.3347238]]
    224 dia. Valores de Entrada -> [1.33467484 1.33465695 1.33464158 1.33462977 1.33462095 1.33461583
     1.33461368 1.33461487 1.33461845 1.33462465 1.33463287 1.33464265
     1.3346535  1.33466542 1.33467758 1.33468938 1.3347007  1.33471119
     1.33472049 1.334728   1.3347342  1.33473885 1.33474147 1.33474255
     1.33474255 1.33474076 1.33473778 1.33473361 1.33472919 1.33472383]
    224 dia. Valor Previsto -> [[1.3347179]]
    225 dia. Valores de Entrada -> [1.33465695 1.33464158 1.33462977 1.33462095 1.33461583 1.33461368
     1.33461487 1.33461845 1.33462465 1.33463287 1.33464265 1.3346535
     1.33466542 1.33467758 1.33468938 1.3347007  1.33471119 1.33472049
     1.334728   1.3347342  1.33473885 1.33474147 1.33474255 1.33474255
     1.33474076 1.33473778 1.33473361 1.33472919 1.33472383 1.33471787]
    225 dia. Valor Previsto -> [[1.3347118]]
    226 dia. Valores de Entrada -> [1.33464158 1.33462977 1.33462095 1.33461583 1.33461368 1.33461487
     1.33461845 1.33462465 1.33463287 1.33464265 1.3346535  1.33466542
     1.33467758 1.33468938 1.3347007  1.33471119 1.33472049 1.334728
     1.3347342  1.33473885 1.33474147 1.33474255 1.33474255 1.33474076
     1.33473778 1.33473361 1.33472919 1.33472383 1.33471787 1.33471179]
    226 dia. Valor Previsto -> [[1.3347055]]
    227 dia. Valores de Entrada -> [1.33462977 1.33462095 1.33461583 1.33461368 1.33461487 1.33461845
     1.33462465 1.33463287 1.33464265 1.3346535  1.33466542 1.33467758
     1.33468938 1.3347007  1.33471119 1.33472049 1.334728   1.3347342
     1.33473885 1.33474147 1.33474255 1.33474255 1.33474076 1.33473778
     1.33473361 1.33472919 1.33472383 1.33471787 1.33471179 1.33470547]
    227 dia. Valor Previsto -> [[1.3346997]]
    228 dia. Valores de Entrada -> [1.33462095 1.33461583 1.33461368 1.33461487 1.33461845 1.33462465
     1.33463287 1.33464265 1.3346535  1.33466542 1.33467758 1.33468938
     1.3347007  1.33471119 1.33472049 1.334728   1.3347342  1.33473885
     1.33474147 1.33474255 1.33474255 1.33474076 1.33473778 1.33473361
     1.33472919 1.33472383 1.33471787 1.33471179 1.33470547 1.33469975]
    228 dia. Valor Previsto -> [[1.3346944]]
    229 dia. Valores de Entrada -> [1.33461583 1.33461368 1.33461487 1.33461845 1.33462465 1.33463287
     1.33464265 1.3346535  1.33466542 1.33467758 1.33468938 1.3347007
     1.33471119 1.33472049 1.334728   1.3347342  1.33473885 1.33474147
     1.33474255 1.33474255 1.33474076 1.33473778 1.33473361 1.33472919
     1.33472383 1.33471787 1.33471179 1.33470547 1.33469975 1.33469439]
    229 dia. Valor Previsto -> [[1.3346899]]
    230 dia. Valores de Entrada -> [1.33461368 1.33461487 1.33461845 1.33462465 1.33463287 1.33464265
     1.3346535  1.33466542 1.33467758 1.33468938 1.3347007  1.33471119
     1.33472049 1.334728   1.3347342  1.33473885 1.33474147 1.33474255
     1.33474255 1.33474076 1.33473778 1.33473361 1.33472919 1.33472383
     1.33471787 1.33471179 1.33470547 1.33469975 1.33469439 1.33468986]
    230 dia. Valor Previsto -> [[1.3346854]]
    231 dia. Valores de Entrada -> [1.33461487 1.33461845 1.33462465 1.33463287 1.33464265 1.3346535
     1.33466542 1.33467758 1.33468938 1.3347007  1.33471119 1.33472049
     1.334728   1.3347342  1.33473885 1.33474147 1.33474255 1.33474255
     1.33474076 1.33473778 1.33473361 1.33472919 1.33472383 1.33471787
     1.33471179 1.33470547 1.33469975 1.33469439 1.33468986 1.33468544]
    231 dia. Valor Previsto -> [[1.3346823]]
    232 dia. Valores de Entrada -> [1.33461845 1.33462465 1.33463287 1.33464265 1.3346535  1.33466542
     1.33467758 1.33468938 1.3347007  1.33471119 1.33472049 1.334728
     1.3347342  1.33473885 1.33474147 1.33474255 1.33474255 1.33474076
     1.33473778 1.33473361 1.33472919 1.33472383 1.33471787 1.33471179
     1.33470547 1.33469975 1.33469439 1.33468986 1.33468544 1.33468235]
    232 dia. Valor Previsto -> [[1.3346801]]
    233 dia. Valores de Entrada -> [1.33462465 1.33463287 1.33464265 1.3346535  1.33466542 1.33467758
     1.33468938 1.3347007  1.33471119 1.33472049 1.334728   1.3347342
     1.33473885 1.33474147 1.33474255 1.33474255 1.33474076 1.33473778
     1.33473361 1.33472919 1.33472383 1.33471787 1.33471179 1.33470547
     1.33469975 1.33469439 1.33468986 1.33468544 1.33468235 1.33468008]
    233 dia. Valor Previsto -> [[1.3346784]]
    234 dia. Valores de Entrada -> [1.33463287 1.33464265 1.3346535  1.33466542 1.33467758 1.33468938
     1.3347007  1.33471119 1.33472049 1.334728   1.3347342  1.33473885
     1.33474147 1.33474255 1.33474255 1.33474076 1.33473778 1.33473361
     1.33472919 1.33472383 1.33471787 1.33471179 1.33470547 1.33469975
     1.33469439 1.33468986 1.33468544 1.33468235 1.33468008 1.33467841]
    234 dia. Valor Previsto -> [[1.3346777]]
    235 dia. Valores de Entrada -> [1.33464265 1.3346535  1.33466542 1.33467758 1.33468938 1.3347007
     1.33471119 1.33472049 1.334728   1.3347342  1.33473885 1.33474147
     1.33474255 1.33474255 1.33474076 1.33473778 1.33473361 1.33472919
     1.33472383 1.33471787 1.33471179 1.33470547 1.33469975 1.33469439
     1.33468986 1.33468544 1.33468235 1.33468008 1.33467841 1.3346777 ]
    235 dia. Valor Previsto -> [[1.3346779]]
    236 dia. Valores de Entrada -> [1.3346535  1.33466542 1.33467758 1.33468938 1.3347007  1.33471119
     1.33472049 1.334728   1.3347342  1.33473885 1.33474147 1.33474255
     1.33474255 1.33474076 1.33473778 1.33473361 1.33472919 1.33472383
     1.33471787 1.33471179 1.33470547 1.33469975 1.33469439 1.33468986
     1.33468544 1.33468235 1.33468008 1.33467841 1.3346777  1.33467793]
    236 dia. Valor Previsto -> [[1.3346789]]
    237 dia. Valores de Entrada -> [1.33466542 1.33467758 1.33468938 1.3347007  1.33471119 1.33472049
     1.334728   1.3347342  1.33473885 1.33474147 1.33474255 1.33474255
     1.33474076 1.33473778 1.33473361 1.33472919 1.33472383 1.33471787
     1.33471179 1.33470547 1.33469975 1.33469439 1.33468986 1.33468544
     1.33468235 1.33468008 1.33467841 1.3346777  1.33467793 1.33467889]
    237 dia. Valor Previsto -> [[1.3346802]]
    238 dia. Valores de Entrada -> [1.33467758 1.33468938 1.3347007  1.33471119 1.33472049 1.334728
     1.3347342  1.33473885 1.33474147 1.33474255 1.33474255 1.33474076
     1.33473778 1.33473361 1.33472919 1.33472383 1.33471787 1.33471179
     1.33470547 1.33469975 1.33469439 1.33468986 1.33468544 1.33468235
     1.33468008 1.33467841 1.3346777  1.33467793 1.33467889 1.3346802 ]
    238 dia. Valor Previsto -> [[1.3346823]]
    239 dia. Valores de Entrada -> [1.33468938 1.3347007  1.33471119 1.33472049 1.334728   1.3347342
     1.33473885 1.33474147 1.33474255 1.33474255 1.33474076 1.33473778
     1.33473361 1.33472919 1.33472383 1.33471787 1.33471179 1.33470547
     1.33469975 1.33469439 1.33468986 1.33468544 1.33468235 1.33468008
     1.33467841 1.3346777  1.33467793 1.33467889 1.3346802  1.33468235]
    239 dia. Valor Previsto -> [[1.3346847]]
    240 dia. Valores de Entrada -> [1.3347007  1.33471119 1.33472049 1.334728   1.3347342  1.33473885
     1.33474147 1.33474255 1.33474255 1.33474076 1.33473778 1.33473361
     1.33472919 1.33472383 1.33471787 1.33471179 1.33470547 1.33469975
     1.33469439 1.33468986 1.33468544 1.33468235 1.33468008 1.33467841
     1.3346777  1.33467793 1.33467889 1.3346802  1.33468235 1.33468473]
    240 dia. Valor Previsto -> [[1.3346877]]
    241 dia. Valores de Entrada -> [1.33471119 1.33472049 1.334728   1.3347342  1.33473885 1.33474147
     1.33474255 1.33474255 1.33474076 1.33473778 1.33473361 1.33472919
     1.33472383 1.33471787 1.33471179 1.33470547 1.33469975 1.33469439
     1.33468986 1.33468544 1.33468235 1.33468008 1.33467841 1.3346777
     1.33467793 1.33467889 1.3346802  1.33468235 1.33468473 1.33468771]
    241 dia. Valor Previsto -> [[1.3346906]]
    242 dia. Valores de Entrada -> [1.33472049 1.334728   1.3347342  1.33473885 1.33474147 1.33474255
     1.33474255 1.33474076 1.33473778 1.33473361 1.33472919 1.33472383
     1.33471787 1.33471179 1.33470547 1.33469975 1.33469439 1.33468986
     1.33468544 1.33468235 1.33468008 1.33467841 1.3346777  1.33467793
     1.33467889 1.3346802  1.33468235 1.33468473 1.33468771 1.33469057]
    242 dia. Valor Previsto -> [[1.3346937]]
    243 dia. Valores de Entrada -> [1.334728   1.3347342  1.33473885 1.33474147 1.33474255 1.33474255
     1.33474076 1.33473778 1.33473361 1.33472919 1.33472383 1.33471787
     1.33471179 1.33470547 1.33469975 1.33469439 1.33468986 1.33468544
     1.33468235 1.33468008 1.33467841 1.3346777  1.33467793 1.33467889
     1.3346802  1.33468235 1.33468473 1.33468771 1.33469057 1.33469367]
    243 dia. Valor Previsto -> [[1.3346967]]
    244 dia. Valores de Entrada -> [1.3347342  1.33473885 1.33474147 1.33474255 1.33474255 1.33474076
     1.33473778 1.33473361 1.33472919 1.33472383 1.33471787 1.33471179
     1.33470547 1.33469975 1.33469439 1.33468986 1.33468544 1.33468235
     1.33468008 1.33467841 1.3346777  1.33467793 1.33467889 1.3346802
     1.33468235 1.33468473 1.33468771 1.33469057 1.33469367 1.33469665]
    244 dia. Valor Previsto -> [[1.3346995]]
    245 dia. Valores de Entrada -> [1.33473885 1.33474147 1.33474255 1.33474255 1.33474076 1.33473778
     1.33473361 1.33472919 1.33472383 1.33471787 1.33471179 1.33470547
     1.33469975 1.33469439 1.33468986 1.33468544 1.33468235 1.33468008
     1.33467841 1.3346777  1.33467793 1.33467889 1.3346802  1.33468235
     1.33468473 1.33468771 1.33469057 1.33469367 1.33469665 1.33469951]
    245 dia. Valor Previsto -> [[1.3347026]]
    246 dia. Valores de Entrada -> [1.33474147 1.33474255 1.33474255 1.33474076 1.33473778 1.33473361
     1.33472919 1.33472383 1.33471787 1.33471179 1.33470547 1.33469975
     1.33469439 1.33468986 1.33468544 1.33468235 1.33468008 1.33467841
     1.3346777  1.33467793 1.33467889 1.3346802  1.33468235 1.33468473
     1.33468771 1.33469057 1.33469367 1.33469665 1.33469951 1.33470261]
    246 dia. Valor Previsto -> [[1.3347046]]
    247 dia. Valores de Entrada -> [1.33474255 1.33474255 1.33474076 1.33473778 1.33473361 1.33472919
     1.33472383 1.33471787 1.33471179 1.33470547 1.33469975 1.33469439
     1.33468986 1.33468544 1.33468235 1.33468008 1.33467841 1.3346777
     1.33467793 1.33467889 1.3346802  1.33468235 1.33468473 1.33468771
     1.33469057 1.33469367 1.33469665 1.33469951 1.33470261 1.33470464]
    247 dia. Valor Previsto -> [[1.334707]]
    248 dia. Valores de Entrada -> [1.33474255 1.33474076 1.33473778 1.33473361 1.33472919 1.33472383
     1.33471787 1.33471179 1.33470547 1.33469975 1.33469439 1.33468986
     1.33468544 1.33468235 1.33468008 1.33467841 1.3346777  1.33467793
     1.33467889 1.3346802  1.33468235 1.33468473 1.33468771 1.33469057
     1.33469367 1.33469665 1.33469951 1.33470261 1.33470464 1.33470702]
    248 dia. Valor Previsto -> [[1.3347086]]
    249 dia. Valores de Entrada -> [1.33474076 1.33473778 1.33473361 1.33472919 1.33472383 1.33471787
     1.33471179 1.33470547 1.33469975 1.33469439 1.33468986 1.33468544
     1.33468235 1.33468008 1.33467841 1.3346777  1.33467793 1.33467889
     1.3346802  1.33468235 1.33468473 1.33468771 1.33469057 1.33469367
     1.33469665 1.33469951 1.33470261 1.33470464 1.33470702 1.33470857]
    249 dia. Valor Previsto -> [[1.3347096]]
    250 dia. Valores de Entrada -> [1.33473778 1.33473361 1.33472919 1.33472383 1.33471787 1.33471179
     1.33470547 1.33469975 1.33469439 1.33468986 1.33468544 1.33468235
     1.33468008 1.33467841 1.3346777  1.33467793 1.33467889 1.3346802
     1.33468235 1.33468473 1.33468771 1.33469057 1.33469367 1.33469665
     1.33469951 1.33470261 1.33470464 1.33470702 1.33470857 1.33470964]
    250 dia. Valor Previsto -> [[1.3347104]]
    251 dia. Valores de Entrada -> [1.33473361 1.33472919 1.33472383 1.33471787 1.33471179 1.33470547
     1.33469975 1.33469439 1.33468986 1.33468544 1.33468235 1.33468008
     1.33467841 1.3346777  1.33467793 1.33467889 1.3346802  1.33468235
     1.33468473 1.33468771 1.33469057 1.33469367 1.33469665 1.33469951
     1.33470261 1.33470464 1.33470702 1.33470857 1.33470964 1.33471036]
    251 dia. Valor Previsto -> [[1.3347108]]
    Previsões -> [[0.7527911067008972], [0.846502423286438], [0.9299150705337524], [1.0117124319076538], [1.0920226573944092], [1.1687005758285522], [1.2388436794281006], [1.3000295162200928], [1.3505486249923706], [1.3899909257888794], [1.4185689687728882], [1.4373146295547485], [1.447493553161621], [1.4501572847366333], [1.4466456174850464], [1.4379850625991821], [1.425490379333496], [1.4098371267318726], [1.3923908472061157], [1.3735653162002563], [1.3540321588516235], [1.3344507217407227], [1.3158483505249023], [1.2987070083618164], [1.2835253477096558], [1.270443081855774], [1.2602202892303467], [1.2527194023132324], [1.2480685710906982], [1.2464756965637207], [1.2478861808776855], [1.2519729137420654], [1.2584737539291382], [1.2670502662658691], [1.2772892713546753], [1.288724422454834], [1.3008631467819214], [1.3132102489471436], [1.325294852256775], [1.3366903066635132], [1.3470335006713867], [1.3560372591018677], [1.363492488861084], [1.369271993637085], [1.3733266592025757], [1.3756746053695679], [1.3763964176177979], [1.3756183385849], [1.3735055923461914], [1.3702503442764282], [1.366061806678772], [1.3611582517623901], [1.355760097503662], [1.3500829935073853], [1.3443313837051392], [1.3386956453323364], [1.3333481550216675], [1.328439474105835], [1.3240960836410522], [1.3204185962677002], [1.3174806833267212], [1.315326452255249], [1.3139735460281372], [1.3134101629257202], [1.3136019706726074], [1.3144891262054443], [1.3159923553466797], [1.3180161714553833], [1.3204540014266968], [1.3231909275054932], [1.326110601425171], [1.3290976285934448], [1.3320436477661133], [1.3348488807678223], [1.3374278545379639], [1.339708924293518], [1.3416374921798706], [1.3431751728057861], [1.344301700592041], [1.3450106382369995], [1.3453130722045898], [1.345231294631958], [1.3447989225387573], [1.344058871269226], [1.3430614471435547], [1.3418594598770142], [1.3405100107192993], [1.3390697240829468], [1.3375937938690186], [1.3361337184906006], [1.334737777709961], [1.3334472179412842], [1.3322970867156982], [1.3313153982162476], [1.3305214643478394], [1.329929232597351], [1.3295420408248901], [1.3293582201004028], [1.3293689489364624], [1.3295583724975586], [1.3299070596694946], [1.3303910493850708], [1.3309838771820068], [1.3316569328308105], [1.3323805332183838], [1.3331273794174194], [1.333869218826294], [1.3345822095870972], [1.3352439403533936], [1.3358358144760132], [1.3363428115844727], [1.336754560470581], [1.3370649814605713], [1.3372712135314941], [1.337373971939087], [1.3373780250549316], [1.3372917175292969], [1.3371243476867676], [1.336888074874878], [1.336596131324768], [1.336262822151184], [1.335902452468872], [1.3355293273925781], [1.3351569175720215], [1.3347975015640259], [1.3344625234603882], [1.3341618776321411], [1.3339022397994995], [1.333689570426941], [1.3335269689559937], [1.3334171772003174], [1.3333584070205688], [1.3333501815795898], [1.3333877325057983], [1.333466649055481], [1.3335813283920288], [1.3337243795394897], [1.3338898420333862], [1.3340699672698975], [1.3342573642730713], [1.3344452381134033], [1.334627389907837], [1.3347976207733154], [1.334951400756836], [1.3350845575332642], [1.3351948261260986], [1.3352794647216797], [1.3353383541107178], [1.3353710174560547], [1.3353785276412964], [1.335362195968628], [1.335325002670288], [1.3352692127227783], [1.3351987600326538], [1.3351167440414429], [1.3350269794464111], [1.3349334001541138], [1.3348387479782104], [1.3347469568252563], [1.3346606492996216], [1.334581971168518], [1.3345136642456055], [1.3344568014144897], [1.334412693977356], [1.334381341934204], [1.3343634605407715], [1.334357738494873], [1.3343647718429565], [1.3343822956085205], [1.3344088792800903], [1.33444344997406], [1.3344842195510864], [1.334529161453247], [1.33457612991333], [1.3346236944198608], [1.334670066833496], [1.3347139358520508], [1.3347539901733398], [1.3347890377044678], [1.3348182439804077], [1.3348416090011597], [1.3348575830459595], [1.3348675966262817], [1.3348711729049683], [1.3348685503005981], [1.3348605632781982], [1.3348474502563477], [1.334830641746521], [1.3348103761672974], [1.3347880840301514], [1.3347647190093994], [1.334740400314331], [1.3347171545028687], [1.3346951007843018], [1.3346748352050781], [1.3346569538116455], [1.3346415758132935], [1.334629774093628], [1.3346209526062012], [1.3346158266067505], [1.3346136808395386], [1.334614872932434], [1.3346184492111206], [1.3346246480941772], [1.3346328735351562], [1.3346426486968994], [1.3346534967422485], [1.3346654176712036], [1.3346775770187378], [1.3346893787384033], [1.3347007036209106], [1.3347111940383911], [1.334720492362976], [1.3347280025482178], [1.3347342014312744], [1.334738850593567], [1.334741473197937], [1.334742546081543], [1.334742546081543], [1.3347407579421997], [1.334737777709961], [1.3347336053848267], [1.3347291946411133], [1.3347238302230835], [1.334717869758606], [1.3347117900848389], [1.3347054719924927], [1.3346997499465942], [1.3346943855285645], [1.3346898555755615], [1.3346854448318481], [1.3346823453903198], [1.3346800804138184], [1.3346784114837646], [1.3346776962280273], [1.3346779346466064], [1.3346788883209229], [1.334680199623108], [1.3346823453903198], [1.3346847295761108], [1.3346877098083496], [1.3346905708312988], [1.3346936702728271], [1.334696650505066], [1.3346995115280151], [1.3347026109695435], [1.3347046375274658], [1.3347070217132568], [1.334708571434021], [1.334709644317627], [1.3347103595733643], [1.3347108364105225]]
    


```python
# Transforma a saída

prev = scaler.inverse_transform(pred_output)
prev = np.array(prev).reshape(1, -1)
list_output_prev = list(prev)
list_output_prev = prev[0].tolist()
list_output_prev

# Pegar as data de previsão

dates = pd.to_datetime(historical_returns.index)
predict_dates = pd.date_range(list(dates)[-1] + pd.DateOffset(1), periods=n_future, freq = 'b').tolist()
predict_dates
```




    [Timestamp('2024-12-27 00:00:00'),
     Timestamp('2024-12-30 00:00:00'),
     Timestamp('2024-12-31 00:00:00'),
     Timestamp('2025-01-01 00:00:00'),
     Timestamp('2025-01-02 00:00:00'),
     Timestamp('2025-01-03 00:00:00'),
     Timestamp('2025-01-06 00:00:00'),
     Timestamp('2025-01-07 00:00:00'),
     Timestamp('2025-01-08 00:00:00'),
     Timestamp('2025-01-09 00:00:00'),
     Timestamp('2025-01-10 00:00:00'),
     Timestamp('2025-01-13 00:00:00'),
     Timestamp('2025-01-14 00:00:00'),
     Timestamp('2025-01-15 00:00:00'),
     Timestamp('2025-01-16 00:00:00'),
     Timestamp('2025-01-17 00:00:00'),
     Timestamp('2025-01-20 00:00:00'),
     Timestamp('2025-01-21 00:00:00'),
     Timestamp('2025-01-22 00:00:00'),
     Timestamp('2025-01-23 00:00:00'),
     Timestamp('2025-01-24 00:00:00'),
     Timestamp('2025-01-27 00:00:00'),
     Timestamp('2025-01-28 00:00:00'),
     Timestamp('2025-01-29 00:00:00'),
     Timestamp('2025-01-30 00:00:00'),
     Timestamp('2025-01-31 00:00:00'),
     Timestamp('2025-02-03 00:00:00'),
     Timestamp('2025-02-04 00:00:00'),
     Timestamp('2025-02-05 00:00:00'),
     Timestamp('2025-02-06 00:00:00'),
     Timestamp('2025-02-07 00:00:00'),
     Timestamp('2025-02-10 00:00:00'),
     Timestamp('2025-02-11 00:00:00'),
     Timestamp('2025-02-12 00:00:00'),
     Timestamp('2025-02-13 00:00:00'),
     Timestamp('2025-02-14 00:00:00'),
     Timestamp('2025-02-17 00:00:00'),
     Timestamp('2025-02-18 00:00:00'),
     Timestamp('2025-02-19 00:00:00'),
     Timestamp('2025-02-20 00:00:00'),
     Timestamp('2025-02-21 00:00:00'),
     Timestamp('2025-02-24 00:00:00'),
     Timestamp('2025-02-25 00:00:00'),
     Timestamp('2025-02-26 00:00:00'),
     Timestamp('2025-02-27 00:00:00'),
     Timestamp('2025-02-28 00:00:00'),
     Timestamp('2025-03-03 00:00:00'),
     Timestamp('2025-03-04 00:00:00'),
     Timestamp('2025-03-05 00:00:00'),
     Timestamp('2025-03-06 00:00:00'),
     Timestamp('2025-03-07 00:00:00'),
     Timestamp('2025-03-10 00:00:00'),
     Timestamp('2025-03-11 00:00:00'),
     Timestamp('2025-03-12 00:00:00'),
     Timestamp('2025-03-13 00:00:00'),
     Timestamp('2025-03-14 00:00:00'),
     Timestamp('2025-03-17 00:00:00'),
     Timestamp('2025-03-18 00:00:00'),
     Timestamp('2025-03-19 00:00:00'),
     Timestamp('2025-03-20 00:00:00'),
     Timestamp('2025-03-21 00:00:00'),
     Timestamp('2025-03-24 00:00:00'),
     Timestamp('2025-03-25 00:00:00'),
     Timestamp('2025-03-26 00:00:00'),
     Timestamp('2025-03-27 00:00:00'),
     Timestamp('2025-03-28 00:00:00'),
     Timestamp('2025-03-31 00:00:00'),
     Timestamp('2025-04-01 00:00:00'),
     Timestamp('2025-04-02 00:00:00'),
     Timestamp('2025-04-03 00:00:00'),
     Timestamp('2025-04-04 00:00:00'),
     Timestamp('2025-04-07 00:00:00'),
     Timestamp('2025-04-08 00:00:00'),
     Timestamp('2025-04-09 00:00:00'),
     Timestamp('2025-04-10 00:00:00'),
     Timestamp('2025-04-11 00:00:00'),
     Timestamp('2025-04-14 00:00:00'),
     Timestamp('2025-04-15 00:00:00'),
     Timestamp('2025-04-16 00:00:00'),
     Timestamp('2025-04-17 00:00:00'),
     Timestamp('2025-04-18 00:00:00'),
     Timestamp('2025-04-21 00:00:00'),
     Timestamp('2025-04-22 00:00:00'),
     Timestamp('2025-04-23 00:00:00'),
     Timestamp('2025-04-24 00:00:00'),
     Timestamp('2025-04-25 00:00:00'),
     Timestamp('2025-04-28 00:00:00'),
     Timestamp('2025-04-29 00:00:00'),
     Timestamp('2025-04-30 00:00:00'),
     Timestamp('2025-05-01 00:00:00'),
     Timestamp('2025-05-02 00:00:00'),
     Timestamp('2025-05-05 00:00:00'),
     Timestamp('2025-05-06 00:00:00'),
     Timestamp('2025-05-07 00:00:00'),
     Timestamp('2025-05-08 00:00:00'),
     Timestamp('2025-05-09 00:00:00'),
     Timestamp('2025-05-12 00:00:00'),
     Timestamp('2025-05-13 00:00:00'),
     Timestamp('2025-05-14 00:00:00'),
     Timestamp('2025-05-15 00:00:00'),
     Timestamp('2025-05-16 00:00:00'),
     Timestamp('2025-05-19 00:00:00'),
     Timestamp('2025-05-20 00:00:00'),
     Timestamp('2025-05-21 00:00:00'),
     Timestamp('2025-05-22 00:00:00'),
     Timestamp('2025-05-23 00:00:00'),
     Timestamp('2025-05-26 00:00:00'),
     Timestamp('2025-05-27 00:00:00'),
     Timestamp('2025-05-28 00:00:00'),
     Timestamp('2025-05-29 00:00:00'),
     Timestamp('2025-05-30 00:00:00'),
     Timestamp('2025-06-02 00:00:00'),
     Timestamp('2025-06-03 00:00:00'),
     Timestamp('2025-06-04 00:00:00'),
     Timestamp('2025-06-05 00:00:00'),
     Timestamp('2025-06-06 00:00:00'),
     Timestamp('2025-06-09 00:00:00'),
     Timestamp('2025-06-10 00:00:00'),
     Timestamp('2025-06-11 00:00:00'),
     Timestamp('2025-06-12 00:00:00'),
     Timestamp('2025-06-13 00:00:00'),
     Timestamp('2025-06-16 00:00:00'),
     Timestamp('2025-06-17 00:00:00'),
     Timestamp('2025-06-18 00:00:00'),
     Timestamp('2025-06-19 00:00:00'),
     Timestamp('2025-06-20 00:00:00'),
     Timestamp('2025-06-23 00:00:00'),
     Timestamp('2025-06-24 00:00:00'),
     Timestamp('2025-06-25 00:00:00'),
     Timestamp('2025-06-26 00:00:00'),
     Timestamp('2025-06-27 00:00:00'),
     Timestamp('2025-06-30 00:00:00'),
     Timestamp('2025-07-01 00:00:00'),
     Timestamp('2025-07-02 00:00:00'),
     Timestamp('2025-07-03 00:00:00'),
     Timestamp('2025-07-04 00:00:00'),
     Timestamp('2025-07-07 00:00:00'),
     Timestamp('2025-07-08 00:00:00'),
     Timestamp('2025-07-09 00:00:00'),
     Timestamp('2025-07-10 00:00:00'),
     Timestamp('2025-07-11 00:00:00'),
     Timestamp('2025-07-14 00:00:00'),
     Timestamp('2025-07-15 00:00:00'),
     Timestamp('2025-07-16 00:00:00'),
     Timestamp('2025-07-17 00:00:00'),
     Timestamp('2025-07-18 00:00:00'),
     Timestamp('2025-07-21 00:00:00'),
     Timestamp('2025-07-22 00:00:00'),
     Timestamp('2025-07-23 00:00:00'),
     Timestamp('2025-07-24 00:00:00'),
     Timestamp('2025-07-25 00:00:00'),
     Timestamp('2025-07-28 00:00:00'),
     Timestamp('2025-07-29 00:00:00'),
     Timestamp('2025-07-30 00:00:00'),
     Timestamp('2025-07-31 00:00:00'),
     Timestamp('2025-08-01 00:00:00'),
     Timestamp('2025-08-04 00:00:00'),
     Timestamp('2025-08-05 00:00:00'),
     Timestamp('2025-08-06 00:00:00'),
     Timestamp('2025-08-07 00:00:00'),
     Timestamp('2025-08-08 00:00:00'),
     Timestamp('2025-08-11 00:00:00'),
     Timestamp('2025-08-12 00:00:00'),
     Timestamp('2025-08-13 00:00:00'),
     Timestamp('2025-08-14 00:00:00'),
     Timestamp('2025-08-15 00:00:00'),
     Timestamp('2025-08-18 00:00:00'),
     Timestamp('2025-08-19 00:00:00'),
     Timestamp('2025-08-20 00:00:00'),
     Timestamp('2025-08-21 00:00:00'),
     Timestamp('2025-08-22 00:00:00'),
     Timestamp('2025-08-25 00:00:00'),
     Timestamp('2025-08-26 00:00:00'),
     Timestamp('2025-08-27 00:00:00'),
     Timestamp('2025-08-28 00:00:00'),
     Timestamp('2025-08-29 00:00:00'),
     Timestamp('2025-09-01 00:00:00'),
     Timestamp('2025-09-02 00:00:00'),
     Timestamp('2025-09-03 00:00:00'),
     Timestamp('2025-09-04 00:00:00'),
     Timestamp('2025-09-05 00:00:00'),
     Timestamp('2025-09-08 00:00:00'),
     Timestamp('2025-09-09 00:00:00'),
     Timestamp('2025-09-10 00:00:00'),
     Timestamp('2025-09-11 00:00:00'),
     Timestamp('2025-09-12 00:00:00'),
     Timestamp('2025-09-15 00:00:00'),
     Timestamp('2025-09-16 00:00:00'),
     Timestamp('2025-09-17 00:00:00'),
     Timestamp('2025-09-18 00:00:00'),
     Timestamp('2025-09-19 00:00:00'),
     Timestamp('2025-09-22 00:00:00'),
     Timestamp('2025-09-23 00:00:00'),
     Timestamp('2025-09-24 00:00:00'),
     Timestamp('2025-09-25 00:00:00'),
     Timestamp('2025-09-26 00:00:00'),
     Timestamp('2025-09-29 00:00:00'),
     Timestamp('2025-09-30 00:00:00'),
     Timestamp('2025-10-01 00:00:00'),
     Timestamp('2025-10-02 00:00:00'),
     Timestamp('2025-10-03 00:00:00'),
     Timestamp('2025-10-06 00:00:00'),
     Timestamp('2025-10-07 00:00:00'),
     Timestamp('2025-10-08 00:00:00'),
     Timestamp('2025-10-09 00:00:00'),
     Timestamp('2025-10-10 00:00:00'),
     Timestamp('2025-10-13 00:00:00'),
     Timestamp('2025-10-14 00:00:00'),
     Timestamp('2025-10-15 00:00:00'),
     Timestamp('2025-10-16 00:00:00'),
     Timestamp('2025-10-17 00:00:00'),
     Timestamp('2025-10-20 00:00:00'),
     Timestamp('2025-10-21 00:00:00'),
     Timestamp('2025-10-22 00:00:00'),
     Timestamp('2025-10-23 00:00:00'),
     Timestamp('2025-10-24 00:00:00'),
     Timestamp('2025-10-27 00:00:00'),
     Timestamp('2025-10-28 00:00:00'),
     Timestamp('2025-10-29 00:00:00'),
     Timestamp('2025-10-30 00:00:00'),
     Timestamp('2025-10-31 00:00:00'),
     Timestamp('2025-11-03 00:00:00'),
     Timestamp('2025-11-04 00:00:00'),
     Timestamp('2025-11-05 00:00:00'),
     Timestamp('2025-11-06 00:00:00'),
     Timestamp('2025-11-07 00:00:00'),
     Timestamp('2025-11-10 00:00:00'),
     Timestamp('2025-11-11 00:00:00'),
     Timestamp('2025-11-12 00:00:00'),
     Timestamp('2025-11-13 00:00:00'),
     Timestamp('2025-11-14 00:00:00'),
     Timestamp('2025-11-17 00:00:00'),
     Timestamp('2025-11-18 00:00:00'),
     Timestamp('2025-11-19 00:00:00'),
     Timestamp('2025-11-20 00:00:00'),
     Timestamp('2025-11-21 00:00:00'),
     Timestamp('2025-11-24 00:00:00'),
     Timestamp('2025-11-25 00:00:00'),
     Timestamp('2025-11-26 00:00:00'),
     Timestamp('2025-11-27 00:00:00'),
     Timestamp('2025-11-28 00:00:00'),
     Timestamp('2025-12-01 00:00:00'),
     Timestamp('2025-12-02 00:00:00'),
     Timestamp('2025-12-03 00:00:00'),
     Timestamp('2025-12-04 00:00:00'),
     Timestamp('2025-12-05 00:00:00'),
     Timestamp('2025-12-08 00:00:00'),
     Timestamp('2025-12-09 00:00:00'),
     Timestamp('2025-12-10 00:00:00'),
     Timestamp('2025-12-11 00:00:00'),
     Timestamp('2025-12-12 00:00:00'),
     Timestamp('2025-12-15 00:00:00')]




```python
# Cria o dataframe de previsão

forecast_dates = []

for i in predict_dates:
    forecast_dates.append(i.date())

df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Saldo':list_output_prev})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

df_forecast = df_forecast.set_index(pd.DatetimeIndex(df_forecast['Date'].values))
df_forecast = df_forecast.drop('Date', axis=1)
df_forecast
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Saldo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-12-27</th>
      <td>142309.550758</td>
    </tr>
    <tr>
      <th>2024-12-30</th>
      <td>143703.718094</td>
    </tr>
    <tr>
      <th>2024-12-31</th>
      <td>144944.669481</td>
    </tr>
    <tr>
      <th>2025-01-01</th>
      <td>146161.589844</td>
    </tr>
    <tr>
      <th>2025-01-02</th>
      <td>147356.385705</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2025-12-09</th>
      <td>150966.863353</td>
    </tr>
    <tr>
      <th>2025-12-10</th>
      <td>150966.886408</td>
    </tr>
    <tr>
      <th>2025-12-11</th>
      <td>150966.902370</td>
    </tr>
    <tr>
      <th>2025-12-12</th>
      <td>150966.913011</td>
    </tr>
    <tr>
      <th>2025-12-15</th>
      <td>150966.920105</td>
    </tr>
  </tbody>
</table>
<p>252 rows × 1 columns</p>
</div>




```python

# Plota o gráfico da previsão
plt.figure(figsize=(12, 6))
plt.plot(portfolio_balance.tail(best_window), label='Histórico')
plt.plot(df_forecast, label='Previsão', color='red')
plt.xlabel('Data')
plt.ylabel('Saldo')
plt.title('Previsão do Saldo do Portfolio')
plt.legend()
plt.grid(True)

# Encontra e plota os valores máximo e mínimo da previsão
#max_value = df_forecast['Saldo'].max()
#min_value = df_forecast['Saldo'].min()
#max_date = df_forecast['Saldo'].idxmax()
#min_date = df_forecast['Saldo'].idxmin()

#plt.scatter(max_date, max_value, color='green', label='Máximo', s=100)
#plt.scatter(min_date, min_value, color='red', label='Mínimo', s=100)

#plt.annotate(f'Máximo: {max_value:.2f}', (max_date, max_value), xytext=(10,10),
             #textcoords='offset points', arrowprops=dict(arrowstyle='->'))
#plt.annotate(f'Mínimo: {min_value:.2f}', (min_date, min_value), xytext=(-80,-20),
             #textcoords='offset points', arrowprops=dict(arrowstyle='->'))

#plt.fill_between(df_forecast.index, df_forecast['Saldo'], min_value, color='orange', alpha=0.3)

plt.legend()
plt.show()

```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_96_0.png)
    


## 6.3  Encontrar Distribuição Ótima do Portfolio

Restrições são condições que durante o processo de otimização precisam ser adereçadas

Nesse caso a restrição é o peso conjunto de todos os ativos, somando o valor igual a 1.

A variável restrição é um dicionário com duas chaves: 'type' e 'fun'

'type' é definido como 'eq', que significa restrição de equidade

'fun' é definido como função chamada check_sum, que vai checar se os pesos dos portfolios somam corretamente o valor '1'.

Bounds são os limites estabelecidos para as variáveis durante o processo de otimização. Nesse caso as variáveis são os pesos do portfolio, e cada peso deve ser entre 0 e 1.


```python
constraints = {'type':'eq','fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.5) for _ in range(len(portfolio))]

# Estabelece pesos iniciais

initial_weights = np.array([1/len(portfolio)]*len(portfolio))
print(initial_weights)
```

    [0.2 0.2 0.2 0.2 0.2]
    


```python
# Otimiza os pesos para maximizar o indicador de Sharpe
# SLSQP é o nome para Sequential Least Squares Quadratic Programming, que é um técnica de otimização numérica muito útil para resolver problemas de otimização não lineares com restrições

optimized_results = minimize(negative_sharpe_ratio, initial_weights, 
                             args=(log_returns, cov_matrix, risk_free_rate), 
                             method='SLSQP', bounds=bounds, 
                             constraints=constraints)

# Estabelece os pesos ótimos para o portfolio

optimal_weights = optimized_results.x
print(optimal_weights)
```

    [4.33162571e-01 3.74027096e-01 4.33247188e-16 1.92810333e-01
     0.00000000e+00]
    

## 6.4 Análise do Portfolio Ótimo

### Informação sobre o Portfólio Ótimo


```python
# Mostra informações do portfolio otimizado

print('Optimal Weights:')
for ticker, weight in zip(portfolio, optimal_weights):
    print(f'{ticker}: {weight:.4f}')
print()

optimal_portfolio_return = expected_returns(optimal_weights, log_returns)
print(f'Optimal Annual Return: {optimal_portfolio_return:.4f}')

optimal_portfolio_volatility = std_deviation(optimal_weights, cov_matrix)
print(f'Expected Volatility: {optimal_portfolio_volatility:.4f}')

optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)
print(f'Sharpe Ratio: {optimal_sharpe_ratio:.4f}')
```

    Optimal Weights:
    PETR4.SA: 0.4332
    SBSP3.SA: 0.3740
    RENT3.SA: 0.0000
    ITUB4.SA: 0.1928
    VALE3.SA: 0.0000
    
    Optimal Annual Return: 0.3096
    Expected Volatility: 0.2202
    Sharpe Ratio: 1.4034
    


```python
# Plota em gráfico a distribuição do portfolio

plt.figure(figsize=(16,9))
plt.bar(portfolio, optimal_weights)
plt.xlabel('Ativos')
plt.ylabel('Pesos')
plt.title('Distribuição do Portfolio otimizado')
plt.show()
```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_104_0.png)
    



```python

# Calcula os retornos históricos ponderados pelo portfolio otimizado
historical_returns_optimized = (log_returns * optimal_weights).sum(axis=1)

# Transforma em uma série
historical_returns_optimized = pd.Series(historical_returns_optimized)

# Exibe os retornos históricos otimizados
print(historical_returns_optimized)

```

    Date
    2021-12-28    0.000000
    2021-12-29   -0.007029
    2021-12-30   -0.001478
    2022-01-03    0.012738
    2022-01-04   -0.003416
                    ...   
    2024-12-18   -0.028202
    2024-12-19   -0.003723
    2024-12-20   -0.003866
    2024-12-23    0.003955
    2024-12-26    0.004815
    Length: 750, dtype: float64
    

### Encontrar o X-Day dos Retornos Históricos do Portfolio Ótimo


```python
# Especificar um intervalo de confiança
confidence_interval = 0.95
test_windows = [30, 60, 90, 180, 252]  # Diferentes janelas de dias

```

### Calcular o VaR (Value at Risk)


```python
# Lista para armazenar os resultados do VaR
test_windows_results_optimal = []

# Loop para calcular o VaR para cada janela de tempo
for window in test_windows:
    range_returns = historical_returns_optimized.rolling(window=window).sum().dropna()

    if range_returns.empty:
        print(f'Janela de {window} dias sem dados suficientes.')
        continue

    VaR = -np.percentile(range_returns, 100 - (confidence_interval * 100)) * portfolio_value
    test_windows_results_optimal.append((window, VaR))
    print(f'\nVaR para janela de {window} dias: R$ {VaR:.2f}\n')

# Mostrar todos os resultados armazenados em test_windows_results
print("\nResultados do VaR para diferentes janelas de tempo:\n")
for window, var in test_windows_results_optimal:
    print(f"Janela de {window} dias: VaR = R$ {var:.2f}")

# Encontrar o melhor valor de VaR (o menor, pois ele representa a perda máxima)
best_window_optimal, best_VaR_optimal = min(test_windows_results_optimal, key=lambda x: x[1])
print(f'\nMelhor janela de tempo: {best_window_optimal} dias com VaR = R$ {best_VaR_optimal:.2f}\n')

# Plotar o gráfico da distribuição dos retornos do portfólio com a melhor janela
range_returns_best_optimized = historical_returns_optimized.rolling(window=best_window_optimal).sum().dropna()
plt.hist(range_returns_best_optimized * portfolio_value, bins=50, density=True, alpha=0.6, color='g')
plt.axvline(-best_VaR_optimal, color='r', linestyle='dashed', linewidth=2, label=f'VaR {confidence_interval*100}% de confiança')
plt.xlabel(f'{best_window_optimal} dias - Retorno do Portfolio (Reais)')
plt.ylabel('Frequência')
plt.title(f'Distribuição dos Retornos do Portfolio - {best_window} dias (Reais)')
plt.legend()
plt.show()
```

    
    VaR para janela de 30 dias: R$ 8970.89
    
    
    VaR para janela de 60 dias: R$ 8310.51
    
    
    VaR para janela de 90 dias: R$ 8532.67
    
    
    VaR para janela de 180 dias: R$ -3562.10
    
    
    VaR para janela de 252 dias: R$ -11953.32
    
    
    Resultados do VaR para diferentes janelas de tempo:
    
    Janela de 30 dias: VaR = R$ 8970.89
    Janela de 60 dias: VaR = R$ 8310.51
    Janela de 90 dias: VaR = R$ 8532.67
    Janela de 180 dias: VaR = R$ -3562.10
    Janela de 252 dias: VaR = R$ -11953.32
    
    Melhor janela de tempo: 252 dias com VaR = R$ -11953.32
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_109_1.png)
    


### Backtesting


```python
# Escolha a melhor janela com base na análise anterior
# # Por exemplo, se a janela de 180 dias for a escolhida

backtesting_days = best_window  # Período de backtesting (1 ano)
violacoes_port_optimized = []

for i in range(backtesting_days, len(historical_returns_optimized)):
    # Calcular o VaR com os dados anteriores ao dia i
    janela_treino = historical_returns_optimized[i-backtesting_days:i]
    range_returns = janela_treino.rolling(window=best_window_optimal).sum().dropna()

    if range_returns.empty:
        print(f'Janela de {best_window_optimal} dias sem dados suficientes no índice {i}.')
        continue  # Pula a iteração se range_returns estiver vazio

    VaR = -np.percentile(range_returns, 100 - (confidence_interval * 100)) * portfolio_value

    # Comparar com o retorno real
    retorno_real = historical_returns_optimized.iloc[i] * portfolio_value
    if retorno_real < -VaR:
        violacoes_port_optimized.append(1)
    else:
        violacoes_port_optimized.append(0)

# Imprimir a taxa de violação observada
taxa_violacao_port_optimized = np.mean(violacoes_port_optimized)
print(f'Taxa de Violação do Portfólio Ótimo Observada: {taxa_violacao_port_optimized:.4f}')
```

    Taxa de Violação do Portfólio Ótimo Observada: 1.0000
    

### Stress Test


```python
# Parâmetros do Stress Test
shock_factor = 0.5  # Aplicando um choque de 50% nos retornos (simula uma queda acentuada)

# Simular cenário de stress aplicando um choque negativo nos retornos
stress_scenario_returns_optimal = historical_returns_optimized.copy()
stress_scenario_returns_optimal *= np.random.uniform(1 - shock_factor, 1 - shock_factor, size=len(stress_scenario_returns_optimal))

# Calcular os retornos acumulados no cenário de stress
range_returns_stress_optimal = stress_scenario_returns_optimal.rolling(window=best_window_optimal).sum().dropna()

# Calcular o VaR no cenário de stress
VaR_stress_optimal = -np.percentile(range_returns_stress_optimal, 100 - (confidence_interval * 100)) * portfolio_value

# Exibir o resultado do VaR em cenário de stress
print(f'\nVaR em cenário de stress (choque de {shock_factor*100}%): R$ {VaR_stress_optimal:.2f}\n\n')

# Visualizar a distribuição dos retornos no cenário de stress
plt.hist(range_returns_stress_optimal * portfolio_value, bins=50, density=True, alpha=0.6, color='r')
plt.axvline(-VaR_stress_optimal, color='b', linestyle='dashed', linewidth=2, label=f'VaR Stress {confidence_interval*100}% de confiança')
plt.xlabel(f'{best_window_optimal} dias - Retorno do Portfolio (Reais) em Stress')
plt.ylabel('Frequência')
plt.title(f'Distribuição dos Retornos do Portfolio - {best_window_optimal} dias (Stress)')
plt.legend()
plt.show()
```

    
    VaR em cenário de stress (choque de 50.0%): R$ -5976.66
    
    
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_113_1.png)
    


### Análise dos Retornos do Portfolio Otimizado


```python
# Calcula o retorno acumulado do portfolio
portfolio_return_optimized = (historical_returns_optimized.cumsum() + 1) * portfolio_value

# Calcula o retorno acumulado do CDI
cdi_return = (cdi['cdi'].cumsum() + 1) * portfolio_value

# Calcula o retorno acumulado do IBOV
ibov_return = (log_ibov['IBOV'].cumsum() + 1) * portfolio_value

# Obter os saldos atuais
saldo_atual_portfolio_optimal = portfolio_return_optimized[-1]
saldo_atual_cdi = cdi_return[-1]
saldo_atual_ibov = ibov_return[-1]

# Calcular as diferenças
diferenca_cdi_optimal = saldo_atual_portfolio_optimal - saldo_atual_cdi
diferenca_ibov_optimal = saldo_atual_portfolio_optimal - saldo_atual_ibov

# Imprimir os resultados
print('\n')
print(f"Saldo atual do Portfolio Otimizado: R$ {saldo_atual_portfolio_optimal:.2f}")
print(f"Saldo atual do CDI: R$ {saldo_atual_cdi:.2f}")
print(f"Saldo atual do IBOV: R$ {saldo_atual_ibov:.2f}\n")
print(f"\nDiferença Portfolio Otimizado vs. CDI: R$ {diferenca_cdi_optimal:.2f}")
print(f"Diferença Portfolio Otimizado vs. IBOV: R$ {diferenca_ibov_optimal:.2f}\n")

# Plota os retornos acumulados
plt.figure(figsize=(12, 6))
plt.plot(historical_returns_optimized.index, portfolio_return_optimized, label='Portfolio Otimizado')
plt.plot(cdi.index, cdi_return, label='CDI', linestyle='--')
plt.plot(log_ibov.index, ibov_return, label='IBOV', linestyle='-.')
plt.axhline(portfolio_value, color='k', linestyle=':', label='Investimento Inicial')
plt.title('Retorno Acumulado do Portfolio Otimizado  vs. Benchmarks (R$)')
plt.xlabel('Data')
plt.ylabel('Valor Acumulado (R$)')
plt.legend()
plt.grid(True)
plt.show()


```

    
    
    Saldo atual do Portfolio Otimizado: R$ 192006.19
    Saldo atual do CDI: R$ 134261.80
    Saldo atual do IBOV: R$ 114376.66
    
    
    Diferença Portfolio Otimizado vs. CDI: R$ 57744.39
    Diferença Portfolio Otimizado vs. IBOV: R$ 77629.54
    
    

    C:\Users\jeand\AppData\Local\Temp\ipykernel_37080\2555195098.py:11: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      saldo_atual_portfolio_optimal = portfolio_return_optimized[-1]
    C:\Users\jeand\AppData\Local\Temp\ipykernel_37080\2555195098.py:12: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      saldo_atual_cdi = cdi_return[-1]
    C:\Users\jeand\AppData\Local\Temp\ipykernel_37080\2555195098.py:13: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      saldo_atual_ibov = ibov_return[-1]
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_115_2.png)
    



```python
# Calcula as métricas para o portfolio
portfolio_mean_return_optimal = historical_returns_optimized.mean() * best_window  # Retorno médio anualizado
portfolio_std_dev_optimal = historical_returns_optimized.std() * np.sqrt(best_window)  # Desvio padrão anualizado
portfolio_sharpe_ratio_optimal = (portfolio_mean_return_optimal - risk_free_rate) / portfolio_std_dev_optimal  # Sharpe Ratio

# Calcula as métricas para o CDI
cdi_mean_return = cdi['cdi'].mean() * best_window
cdi_std_dev = cdi['cdi'].std() * np.sqrt(best_window)
cdi_sharpe_ratio = (cdi_mean_return - risk_free_rate) / cdi_std_dev  # Sharpe Ratio para o CDI (próximo de zero)

# Calcula as métricas para o IBOV
ibov_mean_return = log_ibov['IBOV'].mean() * best_window
ibov_std_dev = log_ibov['IBOV'].std() * np.sqrt(best_window)
ibov_sharpe_ratio = (ibov_mean_return - risk_free_rate) / ibov_std_dev

# Cria um DataFrame para exibir as métricas
metrics_df_optimal = pd.DataFrame({
    'Métrica': ['Retorno Médio Anualizado', 'Desvio Padrão Anualizado', 'Sharpe Ratio'],
    'Portfolio Ótimo': [portfolio_mean_return_optimal, portfolio_std_dev_optimal, portfolio_sharpe_ratio_optimal],
    'CDI': [cdi_mean_return, cdi_std_dev, cdi_sharpe_ratio],
    'IBOV': [ibov_mean_return, ibov_std_dev, ibov_sharpe_ratio]
})

# Exibe o DataFrame
print(metrics_df_optimal.to_string(index=False))

```

                     Métrica  Portfolio Ótimo        CDI     IBOV
    Retorno Médio Anualizado         0.309141   0.114509 0.048370
    Desvio Padrão Anualizado         0.220101   0.000788 0.173979
                Sharpe Ratio         1.402474 144.766572 0.275406
    

## 6.5 Previsão do Portfólio Ótimo

### Portfolio Ótimo - CAPM


```python
# Apaga primeira linha de historical_returns

historical_returns_optimized = historical_returns_optimized.iloc[1:]
historical_returns_optimized

```




    Date
    2021-12-29   -0.007029
    2021-12-30   -0.001478
    2022-01-03    0.012738
    2022-01-04   -0.003416
    2022-01-05   -0.037226
                    ...   
    2024-12-18   -0.028202
    2024-12-19   -0.003723
    2024-12-20   -0.003866
    2024-12-23    0.003955
    2024-12-26    0.004815
    Length: 749, dtype: float64




```python
# Calcula o retorno médio do portfolio
portfolio_mean_return_optimal = historical_returns_optimized.mean() * best_window

# Calcula o retorno médio do IBOV
ibov_mean_return = log_ibov['IBOV'].mean() * best_window

# Calcula a covariância entre o portfolio e o IBOV
cov_portfolio_ibov_optimal = historical_returns_optimized.cov(log_ibov['IBOV']) * best_window

# Calcula a variância do IBOV
var_ibov = log_ibov['IBOV'].var() * best_window

# Calcula o beta do portfolio
beta_portfolio_optimal = cov_portfolio_ibov_optimal / var_ibov

# Calcula o retorno esperado do portfolio usando o CAPM
expected_return_capm_optimal = risk_free_rate + beta_portfolio_optimal * (ibov_mean_return - risk_free_rate)

# Calcula o alfa do portfolio
alpha_portfolio_optimal = portfolio_mean_return_optimal - expected_return_capm_optimal

# Imprime os resultados
print(f"Beta do Portfolio: {beta_portfolio_optimal:.4f}")
print(f"Retorno Esperado (CAPM): {expected_return_capm_optimal:.4f}")
print(f"Alfa do Portfolio: {alpha_portfolio_optimal:.4f}")

```

    Beta do Portfolio: 0.9525
    Retorno Esperado (CAPM): 0.0461
    Alfa do Portfolio: 0.2635
    


```python

# Calcula o erro de previsão do CAPM
capm_prediction_error_optimal = portfolio_mean_return_optimal - expected_return_capm_optimal

# Calcula o R-quadrado do modelo CAPM (quanto da variância do retorno do portfolio é explicada pelo IBOV)
r_squared_capm_optimal = (cov_portfolio_ibov_optimal ** 2) / (var_ibov * historical_returns_optimized.var() * best_window)

# Imprime os resultados adicionais
print(f"Erro de Previsão do CAPM: {capm_prediction_error_optimal:.4f}")
print(f"R-quadrado do Modelo CAPM: {r_squared_capm_optimal:.4f}")

# Interpretação dos resultados
print("\nInterpretação dos Resultados:")
print(f"- Beta do Portfolio ({beta_portfolio_optimal:.4f}): Indica que o portfolio ótimo é {beta_portfolio_optimal:.2f} vezes mais volátil que o IBOV.")
if alpha_portfolio > 0:
    print(f"- Alfa do Portfolio Ótimo ({alpha_portfolio_optimal:.4f}): Positivo, indicando que o portfolio ótimo gerou retornos acima do esperado pelo CAPM, sugerindo uma possível habilidade do gestor.")
elif alpha_portfolio < 0:
    print(f"- Alfa do Portfolio Ótimo ({alpha_portfolio_optimal:.4f}): Negativo, indicando que o portfolio ótimo gerou retornos abaixo do esperado pelo CAPM.")
else:
    print(f"- Alfa do Portfolio Ótimo ({alpha_portfolio_optimal:.4f}): Zero, indicando que o portfolio ótimo gerou retornos em linha com o esperado pelo CAPM.")
print(f"- R-quadrado do Modelo CAPM ({r_squared_capm_optimal:.4f}): Indica que {r_squared_capm_optimal*100:.2f}% da variância do retorno do portfolio ótimo é explicada pelo IBOV.")

# Visualização dos resultados
plt.figure(figsize=(10, 6))
plt.scatter(log_ibov['IBOV'], historical_returns_optimized, alpha=0.6)
plt.xlabel('Retorno do IBOV')
plt.ylabel('Retorno do Portfolio Ótimo')
plt.title('Relação entre Retorno do Portfolio e Retorno do IBOV')

# Adicionar a linha de regressão (CAPM)
x = np.linspace(log_ibov['IBOV'].min(), log_ibov['IBOV'].max(), 100)
y = risk_free_rate + beta_portfolio_optimal * (x - risk_free_rate)
plt.plot(x, y, color='red', label='Linha de Regressão (CAPM)')

plt.legend()
plt.grid(True)
plt.show()
```

    Erro de Previsão do CAPM: 0.2635
    R-quadrado do Modelo CAPM: 0.5661
    
    Interpretação dos Resultados:
    - Beta do Portfolio (0.9525): Indica que o portfolio ótimo é 0.95 vezes mais volátil que o IBOV.
    - Alfa do Portfolio Ótimo (0.2635): Positivo, indicando que o portfolio ótimo gerou retornos acima do esperado pelo CAPM, sugerindo uma possível habilidade do gestor.
    - R-quadrado do Modelo CAPM (0.5661): Indica que 56.61% da variância do retorno do portfolio ótimo é explicada pelo IBOV.
    


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_121_1.png)
    


### Portfolio Ótimo - Regressão Linear


```python
df_modelagem_otima = log_returns.copy()
```


```python

# Transforma historical_returns_optimized em um DataFrame
df_historical_returns_optimized = pd.DataFrame(historical_returns_optimized, columns=['returns'])

# Exibe o DataFrame
print(df_historical_returns_optimized.head())

```

                 returns
    Date                
    2021-12-29 -0.007029
    2021-12-30 -0.001478
    2022-01-03  0.012738
    2022-01-04 -0.003416
    2022-01-05 -0.037226
    


```python
# Junta a df_modelagem, o cdi, e ibov, usando o index de date de df_modelagem como base
df_modelagem_otima = df_modelagem_otima.join(cdi, how='left').join(log_ibov, how='left')
df_modelagem_otima = df_modelagem_otima.join(df_historical_returns_optimized, how = 'left')
df_modelagem_otima.dropna(inplace=True)
```


```python
df_modelagem_otima
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PETR4.SA</th>
      <th>SBSP3.SA</th>
      <th>RENT3.SA</th>
      <th>ITUB4.SA</th>
      <th>VALE3.SA</th>
      <th>cdi</th>
      <th>IBOV</th>
      <th>returns</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-12-29</th>
      <td>-0.008374</td>
      <td>-0.004757</td>
      <td>-0.027451</td>
      <td>-0.008415</td>
      <td>0.002592</td>
      <td>0.000347</td>
      <td>-0.007245</td>
      <td>-0.007029</td>
    </tr>
    <tr>
      <th>2021-12-30</th>
      <td>-0.003158</td>
      <td>0.008247</td>
      <td>0.017127</td>
      <td>-0.016569</td>
      <td>0.009149</td>
      <td>0.000347</td>
      <td>0.006844</td>
      <td>-0.001478</td>
    </tr>
    <tr>
      <th>2022-01-03</th>
      <td>0.022246</td>
      <td>-0.005741</td>
      <td>-0.040626</td>
      <td>0.027222</td>
      <td>0.000513</td>
      <td>0.000347</td>
      <td>-0.008623</td>
      <td>0.012738</td>
    </tr>
    <tr>
      <th>2022-01-04</th>
      <td>0.003774</td>
      <td>-0.027921</td>
      <td>0.005682</td>
      <td>0.027964</td>
      <td>-0.011865</td>
      <td>0.000347</td>
      <td>-0.003934</td>
      <td>-0.003416</td>
    </tr>
    <tr>
      <th>2022-01-05</th>
      <td>-0.039467</td>
      <td>-0.043937</td>
      <td>-0.029344</td>
      <td>-0.019170</td>
      <td>0.009426</td>
      <td>0.000347</td>
      <td>-0.024527</td>
      <td>-0.037226</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2024-12-17</th>
      <td>0.009444</td>
      <td>0.017707</td>
      <td>0.015189</td>
      <td>0.005347</td>
      <td>0.005003</td>
      <td>0.000455</td>
      <td>0.009168</td>
      <td>0.011745</td>
    </tr>
    <tr>
      <th>2024-12-18</th>
      <td>-0.026188</td>
      <td>-0.030144</td>
      <td>-0.027851</td>
      <td>-0.028960</td>
      <td>-0.023441</td>
      <td>0.000455</td>
      <td>-0.031990</td>
      <td>-0.028202</td>
    </tr>
    <tr>
      <th>2024-12-19</th>
      <td>-0.004029</td>
      <td>-0.008110</td>
      <td>0.083916</td>
      <td>0.005474</td>
      <td>-0.019157</td>
      <td>0.000455</td>
      <td>0.003439</td>
      <td>-0.003723</td>
    </tr>
    <tr>
      <th>2024-12-20</th>
      <td>-0.008377</td>
      <td>-0.006559</td>
      <td>0.034127</td>
      <td>0.011494</td>
      <td>0.015684</td>
      <td>0.000455</td>
      <td>0.007514</td>
      <td>-0.003866</td>
    </tr>
    <tr>
      <th>2024-12-23</th>
      <td>0.000271</td>
      <td>0.020341</td>
      <td>-0.024040</td>
      <td>-0.019555</td>
      <td>0.004202</td>
      <td>0.000455</td>
      <td>-0.010994</td>
      <td>0.003955</td>
    </tr>
  </tbody>
</table>
<p>748 rows × 8 columns</p>
</div>




```python
df_modelagem_otima.info()

```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 748 entries, 2021-12-29 to 2024-12-23
    Data columns (total 8 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   PETR4.SA  748 non-null    float64
     1   SBSP3.SA  748 non-null    float64
     2   RENT3.SA  748 non-null    float64
     3   ITUB4.SA  748 non-null    float64
     4   VALE3.SA  748 non-null    float64
     5   cdi       748 non-null    float64
     6   IBOV      748 non-null    float64
     7   returns   748 non-null    float64
    dtypes: float64(8)
    memory usage: 52.6 KB
    


```python
# Previsão Retornos

# Criando uma coluna ordinal para as datas no DataFrame
df_modelagem_otima['Date_ordinal'] = df_modelagem_otima.index.map(lambda x: x.toordinal())

# Executando regressão linear para retornos, IBOV e CDI
resultados_optimal = {}
for coluna in ['returns', 'IBOV', 'cdi']:
    resultados_optimal[coluna] = regressao_linear(df_modelagem_otima, coluna, best_window=best_window)

# Calculando saldo acumulado histórico e futuro para cada série
saldos_optimal = {'historico': {}, 'futuro': {}}
for coluna in ['returns']:
    # Saldo histórico
    saldo_historico_optimal = [valor_investido]
    for retorno in df_modelagem_otima[coluna]:
        saldo_historico_optimal.append(saldo_historico_optimal[-1] * (1 + retorno))
    saldos_optimal['historico'][coluna] = saldo_historico_optimal[1:]
    
    # Saldo futuro
    saldo_futuro_optimal = [saldos_optimal['historico'][coluna][-1]]
    for retorno in resultados_optimal[coluna]['previsoes_futuras']:
        saldo_futuro_optimal.append(saldo_futuro_optimal[-1] * (1 + retorno))
    saldos_optimal['futuro'][coluna] = saldo_futuro_optimal[1:]

# Plotando os resultados
plt.figure(figsize=(14, 8))

# Histórico
for coluna, cor in zip(['returns'], [ 'green']):
    plt.plot(df_modelagem_otima.index, saldos_optimal['historico'][coluna], label=f'Saldo Histórico ({coluna.upper()})', color=cor)

# Histórico
for coluna, cor in zip(['returns', 'IBOV', 'cdi'], ['blue', 'green', 'orange']):
    plt.plot(df_modelagem.index, saldos['historico'][coluna], label=f'Saldo Histórico ({coluna.upper()})', color=cor)

# Futuro
for coluna, cor in zip(['returns', 'IBOV', 'cdi'], ['blue', 'green', 'orange']):
    plt.plot(resultados[coluna]['datas_futuras'], saldos['futuro'][coluna], linestyle='--', label=f'Saldo Previsto ({coluna.upper()})', color=cor)

# Futuro
for coluna, cor in zip(['returns'], ['green']):
    plt.plot(resultados[coluna]['datas_futuras'], saldos_optimal['futuro'][coluna], linestyle='--', label=f'Saldo Previsto ({coluna.upper()})', color=cor)

# Linha do investimento inicial
plt.axhline(y=valor_investido, color='red', linestyle='--', label='Saldo Inicial (Investimento)')

# Configurações do gráfico
plt.xlabel('Data')
plt.ylabel('Saldo (R$)')
plt.title('Comparação de Saldo: Retornos, IBOV e CDI (Histórico e Previsão)')
plt.legend()
plt.grid(True)
plt.show()

# Métricas da regressão
for coluna in ['returns']:
    print(f"\nMétricas para {coluna.upper()}:")
    print(f"  - MSE: {resultados_optimal[coluna]['mse']:.6f}")
    print(f"  - MAE: {resultados_optimal[coluna]['mae']:.6f}")
    print(f"  - R²: {resultados_optimal[coluna]['r2']:.6f}")
    
```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_128_0.png)
    


    
    Métricas para RETURNS:
      - MSE: 0.000192
      - MAE: 0.010315
      - R²: 0.000437
    

### Previsão do Portfólio Ótimo - ARIMA


```python
# Calcula o retorno acumulado do portfolio
portfolio_return_optimized = (historical_returns_optimized.cumsum() + 1) * portfolio_value
print(type(portfolio_return_optimized))
print(portfolio_return_optimized)
portfolio_return_optimized.plot()
```

    <class 'pandas.core.series.Series'>
    Date
    2021-12-29     99297.110720
    2021-12-30     99149.305617
    2022-01-03    100423.074926
    2022-01-04    100081.439507
    2022-01-05     96358.876413
                      ...      
    2024-12-18    191888.033203
    2024-12-19    191515.739627
    2024-12-20    191129.166637
    2024-12-23    191524.699523
    2024-12-26    192006.191557
    Length: 749, dtype: float64
    




    <Axes: xlabel='Date'>




    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_130_2.png)
    



```python
# análise dos retornos

# fazendo a decomposição
decomposition_optimal = seasonal_decompose(historical_returns_optimized.dropna(), model='additive', period=best_window)

# Plotar a decomposição
fig, axes = plt.subplots(4, 1, figsize=(12, 12))
decomposition_optimal.observed.plot(ax=axes[0], title='Observado')
decomposition_optimal.trend.plot(ax=axes[1], title='Tendência')
decomposition_optimal.seasonal.plot(ax=axes[2], title='Sazonalidade')
decomposition_optimal.resid.plot(ax=axes[3], title='Resíduos')
plt.tight_layout()
plt.show()
```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_131_0.png)
    



```python
# Teste de normalidade dos resíduos (Shapiro-Wilk)
statistic_optimal, p_value_optimal = shapiro(decomposition_optimal.resid.dropna())
print(f'\nTeste de Shapiro-Wilk para retornos do portfolio: Estatística={statistic_optimal:.3f}, p-valor={p_value_optimal:.3f}')
if p_value_optimal > 0.05:
    print('Provavelmente normal')
else:
    print('Provavelmente não normal')
```

    
    Teste de Shapiro-Wilk para retornos do portfolio: Estatística=0.983, p-valor=0.000
    Provavelmente não normal
    


```python
# Teste de estacionariedade (Dickey-Fuller Aumentado)
result_adfuller_optimal = adfuller(historical_returns_optimized.dropna())
print(f'\nTeste de Dickey-Fuller Aumentado para retornos do portfolio:')
print('ADF Statistic: %f' % result_adfuller_optimal[0])
print('p-value: %f' % result_adfuller_optimal[1])
print('Critical Values:')
for key, value in result_adfuller_optimal[4].items():
    print('\t%s: %.3f' % (key, value))
if result_adfuller_optimal[1] <= 0.05:
    print('Série provavelmente estacionária\n')
else:
    print('Série provavelmente não estacionária\n')
```

    
    Teste de Dickey-Fuller Aumentado para retornos do portfolio:
    ADF Statistic: -25.672326
    p-value: 0.000000
    Critical Values:
    	1%: -3.439
    	5%: -2.865
    	10%: -2.569
    Série provavelmente estacionária
    
    


```python
# Autocorrelação e Autocorrelação Parcial dos resíduos
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(decomposition_optimal.resid.dropna(), ax=axes[0])
plot_pacf(decomposition_optimal.resid.dropna(), ax=axes[1])
plt.tight_layout()
plt.show()

```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_134_0.png)
    



```python
# Ajustar o modelo Auto-ARIMA
modelo_auto_arima_optimal = auto_arima(historical_returns_optimized,
                               start_p=1, start_q=1,
                               max_p=10, max_q=10,
                               d=0,  # Determina d automaticamente
                               seasonal=True,  # Ajuste conforme necessidade
                               trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)
```

    Performing stepwise search to minimize aic
     ARIMA(1,0,1)(0,0,0)[0] intercept   : AIC=-4278.530, Time=0.61 sec
     ARIMA(0,0,0)(0,0,0)[0] intercept   : AIC=-4279.457, Time=0.15 sec
     ARIMA(1,0,0)(0,0,0)[0] intercept   : AIC=-4280.345, Time=0.10 sec
     ARIMA(0,0,1)(0,0,0)[0] intercept   : AIC=-4280.414, Time=0.16 sec
     ARIMA(0,0,0)(0,0,0)[0]             : AIC=-4275.601, Time=0.07 sec
     ARIMA(0,0,2)(0,0,0)[0] intercept   : AIC=-4278.443, Time=0.35 sec
     ARIMA(1,0,2)(0,0,0)[0] intercept   : AIC=-4276.431, Time=0.48 sec
     ARIMA(0,0,1)(0,0,0)[0]             : AIC=-4277.249, Time=0.07 sec
    
    Best model:  ARIMA(0,0,1)(0,0,0)[0] intercept
    Total fit time: 1.996 seconds
    


```python
print(modelo_auto_arima_optimal.summary())

```

                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                  749
    Model:               SARIMAX(0, 0, 1)   Log Likelihood                2143.207
    Date:                Thu, 26 Dec 2024   AIC                          -4280.414
    Time:                        23:16:08   BIC                          -4266.557
    Sample:                             0   HQIC                         -4275.074
                                    - 749                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    intercept      0.0012      0.001      2.239      0.025       0.000       0.002
    ma.L1          0.0630      0.032      1.944      0.052      -0.001       0.127
    sigma2         0.0002   6.01e-06     31.838      0.000       0.000       0.000
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):               541.43
    Prob(Q):                              1.00   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.33   Skew:                             0.16
    Prob(H) (two-sided):                  0.00   Kurtosis:                         7.15
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    


```python
# Dividir os dados em treino e teste (80% treino, 20% teste)
train_size = int(len(portfolio_return_optimized) * 0.8)
train_data_optimal, test_data_optimal = portfolio_return_optimized[:train_size], portfolio_return_optimized[train_size:]

# Imprimir os shapes dos conjuntos de treino e teste
print("Shape dos dados de treino:", train_data_optimal.shape)
print("Shape dos dados de teste:", test_data_optimal.shape)

```

    Shape dos dados de treino: (599,)
    Shape dos dados de teste: (150,)
    


```python
model_optimal = ARIMA(train_data_optimal, order=(9,1,10))
result_optimal = model_optimal.fit()
result_optimal.summary()
```

    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      self._init_dates(dates, freq)
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\base\model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>y</td>        <th>  No. Observations:  </th>    <td>599</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARIMA(9, 1, 10)</td> <th>  Log Likelihood     </th> <td>-5203.170</td>
</tr>
<tr>
  <th>Date:</th>            <td>Thu, 26 Dec 2024</td> <th>  AIC                </th> <td>10446.341</td>
</tr>
<tr>
  <th>Time:</th>                <td>23:16:14</td>     <th>  BIC                </th> <td>10534.213</td>
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th> <td>10480.553</td>
</tr>
<tr>
  <th></th>                      <td> - 599</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ar.L1</th>  <td>    0.0393</td> <td>    2.649</td> <td>    0.015</td> <td> 0.988</td> <td>   -5.153</td> <td>    5.231</td>
</tr>
<tr>
  <th>ar.L2</th>  <td>   -0.2944</td> <td>    1.103</td> <td>   -0.267</td> <td> 0.790</td> <td>   -2.456</td> <td>    1.868</td>
</tr>
<tr>
  <th>ar.L3</th>  <td>   -0.5794</td> <td>    1.363</td> <td>   -0.425</td> <td> 0.671</td> <td>   -3.250</td> <td>    2.091</td>
</tr>
<tr>
  <th>ar.L4</th>  <td>   -0.0451</td> <td>    2.249</td> <td>   -0.020</td> <td> 0.984</td> <td>   -4.453</td> <td>    4.363</td>
</tr>
<tr>
  <th>ar.L5</th>  <td>   -0.7290</td> <td>    1.148</td> <td>   -0.635</td> <td> 0.525</td> <td>   -2.979</td> <td>    1.521</td>
</tr>
<tr>
  <th>ar.L6</th>  <td>    0.0004</td> <td>    2.435</td> <td>    0.000</td> <td> 1.000</td> <td>   -4.773</td> <td>    4.774</td>
</tr>
<tr>
  <th>ar.L7</th>  <td>   -0.0903</td> <td>    1.116</td> <td>   -0.081</td> <td> 0.936</td> <td>   -2.278</td> <td>    2.098</td>
</tr>
<tr>
  <th>ar.L8</th>  <td>   -0.6146</td> <td>    0.855</td> <td>   -0.719</td> <td> 0.472</td> <td>   -2.289</td> <td>    1.060</td>
</tr>
<tr>
  <th>ar.L9</th>  <td>    0.3596</td> <td>    2.106</td> <td>    0.171</td> <td> 0.864</td> <td>   -3.767</td> <td>    4.486</td>
</tr>
<tr>
  <th>ma.L1</th>  <td>   -0.0263</td> <td>    2.651</td> <td>   -0.010</td> <td> 0.992</td> <td>   -5.223</td> <td>    5.171</td>
</tr>
<tr>
  <th>ma.L2</th>  <td>    0.3123</td> <td>    1.120</td> <td>    0.279</td> <td> 0.780</td> <td>   -1.883</td> <td>    2.508</td>
</tr>
<tr>
  <th>ma.L3</th>  <td>    0.5859</td> <td>    1.435</td> <td>    0.408</td> <td> 0.683</td> <td>   -2.227</td> <td>    3.399</td>
</tr>
<tr>
  <th>ma.L4</th>  <td>    0.0378</td> <td>    2.294</td> <td>    0.016</td> <td> 0.987</td> <td>   -4.458</td> <td>    4.533</td>
</tr>
<tr>
  <th>ma.L5</th>  <td>    0.7292</td> <td>    1.150</td> <td>    0.634</td> <td> 0.526</td> <td>   -1.526</td> <td>    2.984</td>
</tr>
<tr>
  <th>ma.L6</th>  <td>   -0.0100</td> <td>    2.457</td> <td>   -0.004</td> <td> 0.997</td> <td>   -4.825</td> <td>    4.805</td>
</tr>
<tr>
  <th>ma.L7</th>  <td>    0.0916</td> <td>    1.095</td> <td>    0.084</td> <td> 0.933</td> <td>   -2.054</td> <td>    2.237</td>
</tr>
<tr>
  <th>ma.L8</th>  <td>    0.6080</td> <td>    0.858</td> <td>    0.708</td> <td> 0.479</td> <td>   -1.075</td> <td>    2.291</td>
</tr>
<tr>
  <th>ma.L9</th>  <td>   -0.3551</td> <td>    2.091</td> <td>   -0.170</td> <td> 0.865</td> <td>   -4.453</td> <td>    3.742</td>
</tr>
<tr>
  <th>ma.L10</th> <td>   -0.0118</td> <td>    0.038</td> <td>   -0.313</td> <td> 0.755</td> <td>   -0.086</td> <td>    0.062</td>
</tr>
<tr>
  <th>sigma2</th> <td> 2.146e+06</td> <td>  5.6e-05</td> <td> 3.83e+10</td> <td> 0.000</td> <td> 2.15e+06</td> <td> 2.15e+06</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>2.01</td> <th>  Jarque-Bera (JB):  </th> <td>383.47</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.16</td> <th>  Prob(JB):          </th>  <td>0.00</td> 
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.40</td> <th>  Skew:              </th>  <td>0.18</td> 
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td> <th>  Kurtosis:          </th>  <td>6.91</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).<br/>[2] Covariance matrix is singular or near-singular, with condition number 6.23e+26. Standard errors may be unstable.




```python
# Realizar previsões no conjunto de teste
predictions_test_optimal = result_optimal.get_forecast(steps=len(test_data))
pred_conf_int_optimal = predictions_test_optimal.conf_int()
predicted_mean_optimal = predictions_test_optimal.predicted_mean
```

    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
      return get_prediction_index(
    


```python
# Prever 60 dias no futuro
forecast_steps = best_window
forecast_optimal = result_optimal.get_forecast(steps=forecast_steps)
forecast_conf_int_optimal = forecast_optimal.conf_int()
forecast_mean_optimal = forecast_optimal.predicted_mean
```

    c:\Users\jeand\anaconda3\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      return get_prediction_index(
    


```python
# Plotar gráficos

# Gráfico de comparação do teste
plt.figure(figsize=(14, 7))
#plt.plot(portfolio_return.index[:len(train_data)].index, train_data, label ="Treino", color = 'green')
plt.plot(portfolio_return_optimized.index[-len(test_data_optimal):], test_data_optimal, label='Real', color='blue')
plt.plot(portfolio_return_optimized.index[-len(test_data_optimal):], predicted_mean_optimal, label='Previsto (Teste)', color='red')
plt.fill_between(portfolio_return_optimized.index[-len(test_data_optimal):], pred_conf_int_optimal.iloc[:, 0], pred_conf_int_optimal.iloc[:, 1], color='orange', alpha=0.3, label='Intervalo de Confiança (Teste)')
plt.xlabel('Data')
plt.ylabel('Saldo')
plt.title('Comparação entre Real e Previsto no Teste')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_141_0.png)
    



```python
# Criar um índice de datas para a previsão
last_date = portfolio_return_optimized.index[-1]
forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='B')  # 'B' para dias úteis

# Plotar o gráfico da previsão
plt.figure(figsize=(14, 7))
plt.plot(portfolio_return_optimized.index, portfolio_return_optimized, label='Histórico', color='blue')
plt.plot(forecast_index, forecast_mean_optimal, label='Previsão', color='red')
plt.fill_between(forecast_index, forecast_conf_int_optimal.iloc[:, 0], forecast_conf_int_optimal.iloc[:, 1], color='pink', alpha=0.3, label='Intervalo de Confiança')
plt.xlabel('Data')
plt.ylabel('Saldo')
plt.title('Previsão do Retorno do Portfolio Ótimo')
plt.legend()
plt.grid(True)
plt.show()

```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_142_0.png)
    


### Previsão do Portfolio Ótimo - LSTM


```python
# Calcula o retorno acumulado do portfolio
portfolio_return_optimal = (historical_returns_optimized.cumsum() + 1) * portfolio_value

# Cria um DataFrame com o saldo do portfolio por dia
portfolio_balance_optimal = pd.DataFrame({
    'Saldo': portfolio_return_optimal
})

# Exibe o DataFrame
print(portfolio_balance_optimal)

```

                        Saldo
    Date                     
    2021-12-29   99297.110720
    2021-12-30   99149.305617
    2022-01-03  100423.074926
    2022-01-04  100081.439507
    2022-01-05   96358.876413
    ...                   ...
    2024-12-18  191888.033203
    2024-12-19  191515.739627
    2024-12-20  191129.166637
    2024-12-23  191524.699523
    2024-12-26  192006.191557
    
    [749 rows x 1 columns]
    


```python
# Separa linhas de treino e teste
qtd_linhas = len(portfolio_balance_optimal)
qtd_linhas_treino = round(qtd_linhas * 0.7)
qtd_linhas_teste = qtd_linhas - qtd_linhas_treino

info_treino = f'Quantidade de linhas de treino: {qtd_linhas_treino}'
info_teste = f'Quantidade de linhas de teste: {qtd_linhas_teste}'

print(info_treino)
print(info_teste)
```

    Quantidade de linhas de treino: 524
    Quantidade de linhas de teste: 225
    


```python
# Padroniza os dados
scaler = StandardScaler()
df_scaled_optimal = scaler.fit_transform(portfolio_balance_optimal)
```


```python
# Separa os dados em treino e teste
train_optimal = df_scaled_optimal[:qtd_linhas_treino]
test_optimal = df_scaled_optimal[qtd_linhas_treino: qtd_linhas_treino + qtd_linhas_teste]

print(len(train) , len(test))
```

    524 225
    


```python
# Define numero de dias necessários para realizar a previsão do próximo dia
steps = 30

X_train_optimal, Y_train_optimal = create_df(train_optimal, steps )
X_test_optimal, Y_test_optimal = create_df(test_optimal, steps )

print(X_train_optimal.shape)
print(Y_train_optimal.shape)
print(X_test_optimal.shape)
print(Y_test_optimal.shape)
```

    (493, 30)
    (493,)
    (194, 30)
    (194,)
    


```python
# Gerando os dados esperados pelo modelo

X_train_optimal = X_train_optimal.reshape(X_train_optimal.shape[0], X_train_optimal.shape[1], 1)
X_test_optimal = X_test_optimal.reshape(X_test_optimal.shape[0], X_test_optimal.shape[1], 1)

print(X_train_optimal.shape)
print(X_test_optimal.shape)
```

    (493, 30, 1)
    (194, 30, 1)
    


```python
# Montando a rede

model_LSTM_optimal = Sequential()
model_LSTM_optimal.add(LSTM(35, return_sequences=True, input_shape=(steps, 1)))
model_LSTM_optimal.add(LSTM(35, return_sequences=True))
model_LSTM_optimal.add(LSTM(35))
model_LSTM_optimal.add(Dropout(0.2))
model_LSTM_optimal.add(Dense(1))
model_LSTM_optimal.compile(loss='mse', optimizer='adam')

model_LSTM_optimal.summary()
```

    c:\Users\jeand\anaconda3\Lib\site-packages\keras\src\layers\rnn\rnn.py:204: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(**kwargs)
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_14"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ lstm_42 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">35</span>)         │         <span style="color: #00af00; text-decoration-color: #00af00">5,180</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_43 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">35</span>)         │         <span style="color: #00af00; text-decoration-color: #00af00">9,940</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_44 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">35</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">9,940</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_14 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)            │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">35</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_14 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">36</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">25,096</span> (98.03 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">25,096</span> (98.03 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
# Treinamento do Modelo

validation_optimal = model_LSTM_optimal.fit(X_train_optimal, Y_train_optimal, validation_data=(X_test_optimal, Y_test_optimal), epochs=50, batch_size=steps, verbose=2)
```

    Epoch 1/50
    17/17 - 7s - 416ms/step - loss: 0.1466 - val_loss: 0.1256
    Epoch 2/50
    17/17 - 0s - 18ms/step - loss: 0.0420 - val_loss: 0.0317
    Epoch 3/50
    17/17 - 0s - 17ms/step - loss: 0.0342 - val_loss: 0.0932
    Epoch 4/50
    17/17 - 0s - 19ms/step - loss: 0.0298 - val_loss: 0.0312
    Epoch 5/50
    17/17 - 0s - 17ms/step - loss: 0.0321 - val_loss: 0.0441
    Epoch 6/50
    17/17 - 0s - 18ms/step - loss: 0.0270 - val_loss: 0.0206
    Epoch 7/50
    17/17 - 0s - 17ms/step - loss: 0.0274 - val_loss: 0.0125
    Epoch 8/50
    17/17 - 0s - 17ms/step - loss: 0.0255 - val_loss: 0.0110
    Epoch 9/50
    17/17 - 0s - 16ms/step - loss: 0.0254 - val_loss: 0.0285
    Epoch 10/50
    17/17 - 0s - 18ms/step - loss: 0.0239 - val_loss: 0.0114
    Epoch 11/50
    17/17 - 0s - 17ms/step - loss: 0.0254 - val_loss: 0.0127
    Epoch 12/50
    17/17 - 0s - 16ms/step - loss: 0.0224 - val_loss: 0.0122
    Epoch 13/50
    17/17 - 0s - 16ms/step - loss: 0.0221 - val_loss: 0.0113
    Epoch 14/50
    17/17 - 0s - 18ms/step - loss: 0.0230 - val_loss: 0.0102
    Epoch 15/50
    17/17 - 0s - 22ms/step - loss: 0.0205 - val_loss: 0.0122
    Epoch 16/50
    17/17 - 0s - 18ms/step - loss: 0.0206 - val_loss: 0.0153
    Epoch 17/50
    17/17 - 0s - 18ms/step - loss: 0.0185 - val_loss: 0.0133
    Epoch 18/50
    17/17 - 0s - 22ms/step - loss: 0.0206 - val_loss: 0.0137
    Epoch 19/50
    17/17 - 0s - 20ms/step - loss: 0.0175 - val_loss: 0.0171
    Epoch 20/50
    17/17 - 0s - 24ms/step - loss: 0.0189 - val_loss: 0.0141
    Epoch 21/50
    17/17 - 0s - 16ms/step - loss: 0.0175 - val_loss: 0.0114
    Epoch 22/50
    17/17 - 0s - 15ms/step - loss: 0.0172 - val_loss: 0.0162
    Epoch 23/50
    17/17 - 0s - 15ms/step - loss: 0.0157 - val_loss: 0.0118
    Epoch 24/50
    17/17 - 0s - 16ms/step - loss: 0.0184 - val_loss: 0.0097
    Epoch 25/50
    17/17 - 0s - 16ms/step - loss: 0.0172 - val_loss: 0.0156
    Epoch 26/50
    17/17 - 0s - 14ms/step - loss: 0.0166 - val_loss: 0.0144
    Epoch 27/50
    17/17 - 0s - 15ms/step - loss: 0.0165 - val_loss: 0.0093
    Epoch 28/50
    17/17 - 0s - 16ms/step - loss: 0.0157 - val_loss: 0.0124
    Epoch 29/50
    17/17 - 0s - 18ms/step - loss: 0.0154 - val_loss: 0.0173
    Epoch 30/50
    17/17 - 0s - 20ms/step - loss: 0.0152 - val_loss: 0.0188
    Epoch 31/50
    17/17 - 0s - 16ms/step - loss: 0.0200 - val_loss: 0.0118
    Epoch 32/50
    17/17 - 0s - 17ms/step - loss: 0.0165 - val_loss: 0.0100
    Epoch 33/50
    17/17 - 0s - 18ms/step - loss: 0.0140 - val_loss: 0.0160
    Epoch 34/50
    17/17 - 0s - 16ms/step - loss: 0.0139 - val_loss: 0.0104
    Epoch 35/50
    17/17 - 0s - 19ms/step - loss: 0.0128 - val_loss: 0.0133
    Epoch 36/50
    17/17 - 0s - 19ms/step - loss: 0.0145 - val_loss: 0.0142
    Epoch 37/50
    17/17 - 0s - 17ms/step - loss: 0.0141 - val_loss: 0.0094
    Epoch 38/50
    17/17 - 0s - 17ms/step - loss: 0.0129 - val_loss: 0.0125
    Epoch 39/50
    17/17 - 0s - 16ms/step - loss: 0.0144 - val_loss: 0.0092
    Epoch 40/50
    17/17 - 0s - 17ms/step - loss: 0.0141 - val_loss: 0.0179
    Epoch 41/50
    17/17 - 0s - 15ms/step - loss: 0.0143 - val_loss: 0.0146
    Epoch 42/50
    17/17 - 0s - 16ms/step - loss: 0.0145 - val_loss: 0.0213
    Epoch 43/50
    17/17 - 0s - 15ms/step - loss: 0.0140 - val_loss: 0.0158
    Epoch 44/50
    17/17 - 0s - 15ms/step - loss: 0.0137 - val_loss: 0.0088
    Epoch 45/50
    17/17 - 0s - 17ms/step - loss: 0.0127 - val_loss: 0.0097
    Epoch 46/50
    17/17 - 0s - 16ms/step - loss: 0.0127 - val_loss: 0.0120
    Epoch 47/50
    17/17 - 0s - 16ms/step - loss: 0.0129 - val_loss: 0.0146
    Epoch 48/50
    17/17 - 0s - 15ms/step - loss: 0.0112 - val_loss: 0.0140
    Epoch 49/50
    17/17 - 0s - 16ms/step - loss: 0.0145 - val_loss: 0.0094
    Epoch 50/50
    17/17 - 0s - 20ms/step - loss: 0.0118 - val_loss: 0.0147
    


```python
plt.plot(validation_optimal.history['loss'], label='Training Loss')
plt.plot(validation_optimal.history['val_loss'], label='Validation Loss')
plt.title('Erro do Modelo')
plt.ylabel('Erro')
plt.xlabel('Época')
plt.legend()
plt.show()
```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_152_0.png)
    



```python
# Realiza a previsão

prev_optimal = model_LSTM_optimal.predict(X_test_optimal)
prev_optimal = scaler.inverse_transform(prev_optimal)

len_test = len(test_optimal)
len_prev = len(prev_optimal)

print(len_test, len_prev)

days_input_steps = len_test - steps
input_steps = test_optimal[days_input_steps:]
input_steps = np.array(input_steps).reshape(1, -1)
input_steps.shape

# Transformar em lista

list_output_steps = list(input_steps)
list_output_steps = list_output_steps[0].tolist()
list_output_steps
```

    [1m7/7[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 85ms/step
    225 194
    




    [1.3462991138173512,
     1.4111626660721324,
     1.3922706112175796,
     1.4288932008952877,
     1.4574198246402705,
     1.4753281305551478,
     1.4518793939929289,
     1.5251927358245503,
     1.5117304336195494,
     1.5373764614350325,
     1.4932986816937373,
     1.4150100476330063,
     1.4120765056903861,
     1.3934068641514021,
     1.4340850143458495,
     1.4071126845601185,
     1.461965773800224,
     1.393377141190566,
     1.43315390748915,
     1.4575893201548502,
     1.4870038866769013,
     1.4070535262640833,
     1.3855154734823778,
     1.366834992367048,
     1.4056079064613287,
     1.3125035719008107,
     1.3002130152342393,
     1.2874510511861847,
     1.300508808980031,
     1.3164043427753291]




```python
# loop para prever os proximos dias

pred_output_optimal = []
i = 0
n_future = best_window
while(i<n_future):
    if(len(list_output_steps)>steps):
        input_steps = np.array(list_output_steps[1:])
        print(f"{i} dia. Valores de Entrada -> {input_steps}")
        input_steps = input_steps.reshape(1, -1)
        input_steps = input_steps.reshape((1, steps, 1))
        pred = model.predict(input_steps, verbose=0)
        print(f"{i} dia. Valor Previsto -> {pred}")
        list_output_steps.extend(pred[0].tolist())
        list_output_steps = list_output_steps[1:]
        pred_output_optimal.extend(pred.tolist())
        i=i+1
    else:
        input_steps = input_steps.reshape((1, steps, 1))
        pred = model_LSTM_optimal.predict(input_steps, verbose=0)
        print(pred[0])
        list_output_steps.extend(pred[0].tolist())
        print(len(list_output_steps))
        pred_output_optimal.extend(pred.tolist())
        i=i+1
print(f'Previsões -> {pred_output_optimal}')
```

    [1.2769713]
    31
    1 dia. Valores de Entrada -> [1.41116267 1.39227061 1.4288932  1.45741982 1.47532813 1.45187939
     1.52519274 1.51173043 1.53737646 1.49329868 1.41501005 1.41207651
     1.39340686 1.43408501 1.40711268 1.46196577 1.39337714 1.43315391
     1.45758932 1.48700389 1.40705353 1.38551547 1.36683499 1.40560791
     1.31250357 1.30021302 1.28745105 1.30050881 1.31640434 1.27697134]
    1 dia. Valor Previsto -> [[1.2818434]]
    2 dia. Valores de Entrada -> [1.39227061 1.4288932  1.45741982 1.47532813 1.45187939 1.52519274
     1.51173043 1.53737646 1.49329868 1.41501005 1.41207651 1.39340686
     1.43408501 1.40711268 1.46196577 1.39337714 1.43315391 1.45758932
     1.48700389 1.40705353 1.38551547 1.36683499 1.40560791 1.31250357
     1.30021302 1.28745105 1.30050881 1.31640434 1.27697134 1.28184342]
    2 dia. Valor Previsto -> [[1.279135]]
    3 dia. Valores de Entrada -> [1.4288932  1.45741982 1.47532813 1.45187939 1.52519274 1.51173043
     1.53737646 1.49329868 1.41501005 1.41207651 1.39340686 1.43408501
     1.40711268 1.46196577 1.39337714 1.43315391 1.45758932 1.48700389
     1.40705353 1.38551547 1.36683499 1.40560791 1.31250357 1.30021302
     1.28745105 1.30050881 1.31640434 1.27697134 1.28184342 1.27913499]
    3 dia. Valor Previsto -> [[1.2785907]]
    4 dia. Valores de Entrada -> [1.45741982 1.47532813 1.45187939 1.52519274 1.51173043 1.53737646
     1.49329868 1.41501005 1.41207651 1.39340686 1.43408501 1.40711268
     1.46196577 1.39337714 1.43315391 1.45758932 1.48700389 1.40705353
     1.38551547 1.36683499 1.40560791 1.31250357 1.30021302 1.28745105
     1.30050881 1.31640434 1.27697134 1.28184342 1.27913499 1.27859068]
    4 dia. Valor Previsto -> [[1.2800685]]
    5 dia. Valores de Entrada -> [1.47532813 1.45187939 1.52519274 1.51173043 1.53737646 1.49329868
     1.41501005 1.41207651 1.39340686 1.43408501 1.40711268 1.46196577
     1.39337714 1.43315391 1.45758932 1.48700389 1.40705353 1.38551547
     1.36683499 1.40560791 1.31250357 1.30021302 1.28745105 1.30050881
     1.31640434 1.27697134 1.28184342 1.27913499 1.27859068 1.28006852]
    5 dia. Valor Previsto -> [[1.283322]]
    6 dia. Valores de Entrada -> [1.45187939 1.52519274 1.51173043 1.53737646 1.49329868 1.41501005
     1.41207651 1.39340686 1.43408501 1.40711268 1.46196577 1.39337714
     1.43315391 1.45758932 1.48700389 1.40705353 1.38551547 1.36683499
     1.40560791 1.31250357 1.30021302 1.28745105 1.30050881 1.31640434
     1.27697134 1.28184342 1.27913499 1.27859068 1.28006852 1.28332198]
    6 dia. Valor Previsto -> [[1.288106]]
    7 dia. Valores de Entrada -> [1.52519274 1.51173043 1.53737646 1.49329868 1.41501005 1.41207651
     1.39340686 1.43408501 1.40711268 1.46196577 1.39337714 1.43315391
     1.45758932 1.48700389 1.40705353 1.38551547 1.36683499 1.40560791
     1.31250357 1.30021302 1.28745105 1.30050881 1.31640434 1.27697134
     1.28184342 1.27913499 1.27859068 1.28006852 1.28332198 1.28810596]
    7 dia. Valor Previsto -> [[1.2942042]]
    8 dia. Valores de Entrada -> [1.51173043 1.53737646 1.49329868 1.41501005 1.41207651 1.39340686
     1.43408501 1.40711268 1.46196577 1.39337714 1.43315391 1.45758932
     1.48700389 1.40705353 1.38551547 1.36683499 1.40560791 1.31250357
     1.30021302 1.28745105 1.30050881 1.31640434 1.27697134 1.28184342
     1.27913499 1.27859068 1.28006852 1.28332198 1.28810596 1.29420424]
    8 dia. Valor Previsto -> [[1.3011442]]
    9 dia. Valores de Entrada -> [1.53737646 1.49329868 1.41501005 1.41207651 1.39340686 1.43408501
     1.40711268 1.46196577 1.39337714 1.43315391 1.45758932 1.48700389
     1.40705353 1.38551547 1.36683499 1.40560791 1.31250357 1.30021302
     1.28745105 1.30050881 1.31640434 1.27697134 1.28184342 1.27913499
     1.27859068 1.28006852 1.28332198 1.28810596 1.29420424 1.30114424]
    9 dia. Valor Previsto -> [[1.3086917]]
    10 dia. Valores de Entrada -> [1.49329868 1.41501005 1.41207651 1.39340686 1.43408501 1.40711268
     1.46196577 1.39337714 1.43315391 1.45758932 1.48700389 1.40705353
     1.38551547 1.36683499 1.40560791 1.31250357 1.30021302 1.28745105
     1.30050881 1.31640434 1.27697134 1.28184342 1.27913499 1.27859068
     1.28006852 1.28332198 1.28810596 1.29420424 1.30114424 1.30869174]
    10 dia. Valor Previsto -> [[1.3164598]]
    11 dia. Valores de Entrada -> [1.41501005 1.41207651 1.39340686 1.43408501 1.40711268 1.46196577
     1.39337714 1.43315391 1.45758932 1.48700389 1.40705353 1.38551547
     1.36683499 1.40560791 1.31250357 1.30021302 1.28745105 1.30050881
     1.31640434 1.27697134 1.28184342 1.27913499 1.27859068 1.28006852
     1.28332198 1.28810596 1.29420424 1.30114424 1.30869174 1.31645977]
    11 dia. Valor Previsto -> [[1.3242517]]
    12 dia. Valores de Entrada -> [1.41207651 1.39340686 1.43408501 1.40711268 1.46196577 1.39337714
     1.43315391 1.45758932 1.48700389 1.40705353 1.38551547 1.36683499
     1.40560791 1.31250357 1.30021302 1.28745105 1.30050881 1.31640434
     1.27697134 1.28184342 1.27913499 1.27859068 1.28006852 1.28332198
     1.28810596 1.29420424 1.30114424 1.30869174 1.31645977 1.32425165]
    12 dia. Valor Previsto -> [[1.331899]]
    13 dia. Valores de Entrada -> [1.39340686 1.43408501 1.40711268 1.46196577 1.39337714 1.43315391
     1.45758932 1.48700389 1.40705353 1.38551547 1.36683499 1.40560791
     1.31250357 1.30021302 1.28745105 1.30050881 1.31640434 1.27697134
     1.28184342 1.27913499 1.27859068 1.28006852 1.28332198 1.28810596
     1.29420424 1.30114424 1.30869174 1.31645977 1.32425165 1.33189905]
    13 dia. Valor Previsto -> [[1.3390741]]
    14 dia. Valores de Entrada -> [1.43408501 1.40711268 1.46196577 1.39337714 1.43315391 1.45758932
     1.48700389 1.40705353 1.38551547 1.36683499 1.40560791 1.31250357
     1.30021302 1.28745105 1.30050881 1.31640434 1.27697134 1.28184342
     1.27913499 1.27859068 1.28006852 1.28332198 1.28810596 1.29420424
     1.30114424 1.30869174 1.31645977 1.32425165 1.33189905 1.33907413]
    14 dia. Valor Previsto -> [[1.3455925]]
    15 dia. Valores de Entrada -> [1.40711268 1.46196577 1.39337714 1.43315391 1.45758932 1.48700389
     1.40705353 1.38551547 1.36683499 1.40560791 1.31250357 1.30021302
     1.28745105 1.30050881 1.31640434 1.27697134 1.28184342 1.27913499
     1.27859068 1.28006852 1.28332198 1.28810596 1.29420424 1.30114424
     1.30869174 1.31645977 1.32425165 1.33189905 1.33907413 1.3455925 ]
    15 dia. Valor Previsto -> [[1.3511778]]
    16 dia. Valores de Entrada -> [1.46196577 1.39337714 1.43315391 1.45758932 1.48700389 1.40705353
     1.38551547 1.36683499 1.40560791 1.31250357 1.30021302 1.28745105
     1.30050881 1.31640434 1.27697134 1.28184342 1.27913499 1.27859068
     1.28006852 1.28332198 1.28810596 1.29420424 1.30114424 1.30869174
     1.31645977 1.32425165 1.33189905 1.33907413 1.3455925  1.35117781]
    16 dia. Valor Previsto -> [[1.3557976]]
    17 dia. Valores de Entrada -> [1.39337714 1.43315391 1.45758932 1.48700389 1.40705353 1.38551547
     1.36683499 1.40560791 1.31250357 1.30021302 1.28745105 1.30050881
     1.31640434 1.27697134 1.28184342 1.27913499 1.27859068 1.28006852
     1.28332198 1.28810596 1.29420424 1.30114424 1.30869174 1.31645977
     1.32425165 1.33189905 1.33907413 1.3455925  1.35117781 1.35579765]
    17 dia. Valor Previsto -> [[1.3592373]]
    18 dia. Valores de Entrada -> [1.43315391 1.45758932 1.48700389 1.40705353 1.38551547 1.36683499
     1.40560791 1.31250357 1.30021302 1.28745105 1.30050881 1.31640434
     1.27697134 1.28184342 1.27913499 1.27859068 1.28006852 1.28332198
     1.28810596 1.29420424 1.30114424 1.30869174 1.31645977 1.32425165
     1.33189905 1.33907413 1.3455925  1.35117781 1.35579765 1.35923731]
    18 dia. Valor Previsto -> [[1.3616556]]
    19 dia. Valores de Entrada -> [1.45758932 1.48700389 1.40705353 1.38551547 1.36683499 1.40560791
     1.31250357 1.30021302 1.28745105 1.30050881 1.31640434 1.27697134
     1.28184342 1.27913499 1.27859068 1.28006852 1.28332198 1.28810596
     1.29420424 1.30114424 1.30869174 1.31645977 1.32425165 1.33189905
     1.33907413 1.3455925  1.35117781 1.35579765 1.35923731 1.36165559]
    19 dia. Valor Previsto -> [[1.3629122]]
    20 dia. Valores de Entrada -> [1.48700389 1.40705353 1.38551547 1.36683499 1.40560791 1.31250357
     1.30021302 1.28745105 1.30050881 1.31640434 1.27697134 1.28184342
     1.27913499 1.27859068 1.28006852 1.28332198 1.28810596 1.29420424
     1.30114424 1.30869174 1.31645977 1.32425165 1.33189905 1.33907413
     1.3455925  1.35117781 1.35579765 1.35923731 1.36165559 1.36291218]
    20 dia. Valor Previsto -> [[1.3630438]]
    21 dia. Valores de Entrada -> [1.40705353 1.38551547 1.36683499 1.40560791 1.31250357 1.30021302
     1.28745105 1.30050881 1.31640434 1.27697134 1.28184342 1.27913499
     1.27859068 1.28006852 1.28332198 1.28810596 1.29420424 1.30114424
     1.30869174 1.31645977 1.32425165 1.33189905 1.33907413 1.3455925
     1.35117781 1.35579765 1.35923731 1.36165559 1.36291218 1.36304379]
    21 dia. Valor Previsto -> [[1.3620915]]
    22 dia. Valores de Entrada -> [1.38551547 1.36683499 1.40560791 1.31250357 1.30021302 1.28745105
     1.30050881 1.31640434 1.27697134 1.28184342 1.27913499 1.27859068
     1.28006852 1.28332198 1.28810596 1.29420424 1.30114424 1.30869174
     1.31645977 1.32425165 1.33189905 1.33907413 1.3455925  1.35117781
     1.35579765 1.35923731 1.36165559 1.36291218 1.36304379 1.36209154]
    22 dia. Valor Previsto -> [[1.3603452]]
    23 dia. Valores de Entrada -> [1.36683499 1.40560791 1.31250357 1.30021302 1.28745105 1.30050881
     1.31640434 1.27697134 1.28184342 1.27913499 1.27859068 1.28006852
     1.28332198 1.28810596 1.29420424 1.30114424 1.30869174 1.31645977
     1.32425165 1.33189905 1.33907413 1.3455925  1.35117781 1.35579765
     1.35923731 1.36165559 1.36291218 1.36304379 1.36209154 1.36034524]
    23 dia. Valor Previsto -> [[1.3578831]]
    24 dia. Valores de Entrada -> [1.40560791 1.31250357 1.30021302 1.28745105 1.30050881 1.31640434
     1.27697134 1.28184342 1.27913499 1.27859068 1.28006852 1.28332198
     1.28810596 1.29420424 1.30114424 1.30869174 1.31645977 1.32425165
     1.33189905 1.33907413 1.3455925  1.35117781 1.35579765 1.35923731
     1.36165559 1.36291218 1.36304379 1.36209154 1.36034524 1.3578831 ]
    24 dia. Valor Previsto -> [[1.3548479]]
    25 dia. Valores de Entrada -> [1.31250357 1.30021302 1.28745105 1.30050881 1.31640434 1.27697134
     1.28184342 1.27913499 1.27859068 1.28006852 1.28332198 1.28810596
     1.29420424 1.30114424 1.30869174 1.31645977 1.32425165 1.33189905
     1.33907413 1.3455925  1.35117781 1.35579765 1.35923731 1.36165559
     1.36291218 1.36304379 1.36209154 1.36034524 1.3578831  1.35484791]
    25 dia. Valor Previsto -> [[1.3512769]]
    26 dia. Valores de Entrada -> [1.30021302 1.28745105 1.30050881 1.31640434 1.27697134 1.28184342
     1.27913499 1.27859068 1.28006852 1.28332198 1.28810596 1.29420424
     1.30114424 1.30869174 1.31645977 1.32425165 1.33189905 1.33907413
     1.3455925  1.35117781 1.35579765 1.35923731 1.36165559 1.36291218
     1.36304379 1.36209154 1.36034524 1.3578831  1.35484791 1.35127687]
    26 dia. Valor Previsto -> [[1.347534]]
    27 dia. Valores de Entrada -> [1.28745105 1.30050881 1.31640434 1.27697134 1.28184342 1.27913499
     1.27859068 1.28006852 1.28332198 1.28810596 1.29420424 1.30114424
     1.30869174 1.31645977 1.32425165 1.33189905 1.33907413 1.3455925
     1.35117781 1.35579765 1.35923731 1.36165559 1.36291218 1.36304379
     1.36209154 1.36034524 1.3578831  1.35484791 1.35127687 1.34753394]
    27 dia. Valor Previsto -> [[1.3436749]]
    28 dia. Valores de Entrada -> [1.30050881 1.31640434 1.27697134 1.28184342 1.27913499 1.27859068
     1.28006852 1.28332198 1.28810596 1.29420424 1.30114424 1.30869174
     1.31645977 1.32425165 1.33189905 1.33907413 1.3455925  1.35117781
     1.35579765 1.35923731 1.36165559 1.36291218 1.36304379 1.36209154
     1.36034524 1.3578831  1.35484791 1.35127687 1.34753394 1.3436749 ]
    28 dia. Valor Previsto -> [[1.3398391]]
    29 dia. Valores de Entrada -> [1.31640434 1.27697134 1.28184342 1.27913499 1.27859068 1.28006852
     1.28332198 1.28810596 1.29420424 1.30114424 1.30869174 1.31645977
     1.32425165 1.33189905 1.33907413 1.3455925  1.35117781 1.35579765
     1.35923731 1.36165559 1.36291218 1.36304379 1.36209154 1.36034524
     1.3578831  1.35484791 1.35127687 1.34753394 1.3436749  1.3398391 ]
    29 dia. Valor Previsto -> [[1.3361022]]
    30 dia. Valores de Entrada -> [1.27697134 1.28184342 1.27913499 1.27859068 1.28006852 1.28332198
     1.28810596 1.29420424 1.30114424 1.30869174 1.31645977 1.32425165
     1.33189905 1.33907413 1.3455925  1.35117781 1.35579765 1.35923731
     1.36165559 1.36291218 1.36304379 1.36209154 1.36034524 1.3578831
     1.35484791 1.35127687 1.34753394 1.3436749  1.3398391  1.33610225]
    30 dia. Valor Previsto -> [[1.332551]]
    31 dia. Valores de Entrada -> [1.28184342 1.27913499 1.27859068 1.28006852 1.28332198 1.28810596
     1.29420424 1.30114424 1.30869174 1.31645977 1.32425165 1.33189905
     1.33907413 1.3455925  1.35117781 1.35579765 1.35923731 1.36165559
     1.36291218 1.36304379 1.36209154 1.36034524 1.3578831  1.35484791
     1.35127687 1.34753394 1.3436749  1.3398391  1.33610225 1.332551  ]
    31 dia. Valor Previsto -> [[1.3293762]]
    32 dia. Valores de Entrada -> [1.27913499 1.27859068 1.28006852 1.28332198 1.28810596 1.29420424
     1.30114424 1.30869174 1.31645977 1.32425165 1.33189905 1.33907413
     1.3455925  1.35117781 1.35579765 1.35923731 1.36165559 1.36291218
     1.36304379 1.36209154 1.36034524 1.3578831  1.35484791 1.35127687
     1.34753394 1.3436749  1.3398391  1.33610225 1.332551   1.32937622]
    32 dia. Valor Previsto -> [[1.3266032]]
    33 dia. Valores de Entrada -> [1.27859068 1.28006852 1.28332198 1.28810596 1.29420424 1.30114424
     1.30869174 1.31645977 1.32425165 1.33189905 1.33907413 1.3455925
     1.35117781 1.35579765 1.35923731 1.36165559 1.36291218 1.36304379
     1.36209154 1.36034524 1.3578831  1.35484791 1.35127687 1.34753394
     1.3436749  1.3398391  1.33610225 1.332551   1.32937622 1.32660317]
    33 dia. Valor Previsto -> [[1.3243059]]
    34 dia. Valores de Entrada -> [1.28006852 1.28332198 1.28810596 1.29420424 1.30114424 1.30869174
     1.31645977 1.32425165 1.33189905 1.33907413 1.3455925  1.35117781
     1.35579765 1.35923731 1.36165559 1.36291218 1.36304379 1.36209154
     1.36034524 1.3578831  1.35484791 1.35127687 1.34753394 1.3436749
     1.3398391  1.33610225 1.332551   1.32937622 1.32660317 1.32430589]
    34 dia. Valor Previsto -> [[1.3225249]]
    35 dia. Valores de Entrada -> [1.28332198 1.28810596 1.29420424 1.30114424 1.30869174 1.31645977
     1.32425165 1.33189905 1.33907413 1.3455925  1.35117781 1.35579765
     1.35923731 1.36165559 1.36291218 1.36304379 1.36209154 1.36034524
     1.3578831  1.35484791 1.35127687 1.34753394 1.3436749  1.3398391
     1.33610225 1.332551   1.32937622 1.32660317 1.32430589 1.32252491]
    35 dia. Valor Previsto -> [[1.3212831]]
    36 dia. Valores de Entrada -> [1.28810596 1.29420424 1.30114424 1.30869174 1.31645977 1.32425165
     1.33189905 1.33907413 1.3455925  1.35117781 1.35579765 1.35923731
     1.36165559 1.36291218 1.36304379 1.36209154 1.36034524 1.3578831
     1.35484791 1.35127687 1.34753394 1.3436749  1.3398391  1.33610225
     1.332551   1.32937622 1.32660317 1.32430589 1.32252491 1.3212831 ]
    36 dia. Valor Previsto -> [[1.3205829]]
    37 dia. Valores de Entrada -> [1.29420424 1.30114424 1.30869174 1.31645977 1.32425165 1.33189905
     1.33907413 1.3455925  1.35117781 1.35579765 1.35923731 1.36165559
     1.36291218 1.36304379 1.36209154 1.36034524 1.3578831  1.35484791
     1.35127687 1.34753394 1.3436749  1.3398391  1.33610225 1.332551
     1.32937622 1.32660317 1.32430589 1.32252491 1.3212831  1.32058287]
    37 dia. Valor Previsto -> [[1.3204082]]
    38 dia. Valores de Entrada -> [1.30114424 1.30869174 1.31645977 1.32425165 1.33189905 1.33907413
     1.3455925  1.35117781 1.35579765 1.35923731 1.36165559 1.36291218
     1.36304379 1.36209154 1.36034524 1.3578831  1.35484791 1.35127687
     1.34753394 1.3436749  1.3398391  1.33610225 1.332551   1.32937622
     1.32660317 1.32430589 1.32252491 1.3212831  1.32058287 1.32040823]
    38 dia. Valor Previsto -> [[1.3207276]]
    39 dia. Valores de Entrada -> [1.30869174 1.31645977 1.32425165 1.33189905 1.33907413 1.3455925
     1.35117781 1.35579765 1.35923731 1.36165559 1.36291218 1.36304379
     1.36209154 1.36034524 1.3578831  1.35484791 1.35127687 1.34753394
     1.3436749  1.3398391  1.33610225 1.332551   1.32937622 1.32660317
     1.32430589 1.32252491 1.3212831  1.32058287 1.32040823 1.32072759]
    39 dia. Valor Previsto -> [[1.3214936]]
    40 dia. Valores de Entrada -> [1.31645977 1.32425165 1.33189905 1.33907413 1.3455925  1.35117781
     1.35579765 1.35923731 1.36165559 1.36291218 1.36304379 1.36209154
     1.36034524 1.3578831  1.35484791 1.35127687 1.34753394 1.3436749
     1.3398391  1.33610225 1.332551   1.32937622 1.32660317 1.32430589
     1.32252491 1.3212831  1.32058287 1.32040823 1.32072759 1.32149363]
    40 dia. Valor Previsto -> [[1.3226484]]
    41 dia. Valores de Entrada -> [1.32425165 1.33189905 1.33907413 1.3455925  1.35117781 1.35579765
     1.35923731 1.36165559 1.36291218 1.36304379 1.36209154 1.36034524
     1.3578831  1.35484791 1.35127687 1.34753394 1.3436749  1.3398391
     1.33610225 1.332551   1.32937622 1.32660317 1.32430589 1.32252491
     1.3212831  1.32058287 1.32040823 1.32072759 1.32149363 1.32264841]
    41 dia. Valor Previsto -> [[1.3241237]]
    42 dia. Valores de Entrada -> [1.33189905 1.33907413 1.3455925  1.35117781 1.35579765 1.35923731
     1.36165559 1.36291218 1.36304379 1.36209154 1.36034524 1.3578831
     1.35484791 1.35127687 1.34753394 1.3436749  1.3398391  1.33610225
     1.332551   1.32937622 1.32660317 1.32430589 1.32252491 1.3212831
     1.32058287 1.32040823 1.32072759 1.32149363 1.32264841 1.32412374]
    42 dia. Valor Previsto -> [[1.3258461]]
    43 dia. Valores de Entrada -> [1.33907413 1.3455925  1.35117781 1.35579765 1.35923731 1.36165559
     1.36291218 1.36304379 1.36209154 1.36034524 1.3578831  1.35484791
     1.35127687 1.34753394 1.3436749  1.3398391  1.33610225 1.332551
     1.32937622 1.32660317 1.32430589 1.32252491 1.3212831  1.32058287
     1.32040823 1.32072759 1.32149363 1.32264841 1.32412374 1.32584608]
    43 dia. Valor Previsto -> [[1.3277376]]
    44 dia. Valores de Entrada -> [1.3455925  1.35117781 1.35579765 1.35923731 1.36165559 1.36291218
     1.36304379 1.36209154 1.36034524 1.3578831  1.35484791 1.35127687
     1.34753394 1.3436749  1.3398391  1.33610225 1.332551   1.32937622
     1.32660317 1.32430589 1.32252491 1.3212831  1.32058287 1.32040823
     1.32072759 1.32149363 1.32264841 1.32412374 1.32584608 1.32773757]
    44 dia. Valor Previsto -> [[1.3297203]]
    45 dia. Valores de Entrada -> [1.35117781 1.35579765 1.35923731 1.36165559 1.36291218 1.36304379
     1.36209154 1.36034524 1.3578831  1.35484791 1.35127687 1.34753394
     1.3436749  1.3398391  1.33610225 1.332551   1.32937622 1.32660317
     1.32430589 1.32252491 1.3212831  1.32058287 1.32040823 1.32072759
     1.32149363 1.32264841 1.32412374 1.32584608 1.32773757 1.32972026]
    45 dia. Valor Previsto -> [[1.3317195]]
    46 dia. Valores de Entrada -> [1.35579765 1.35923731 1.36165559 1.36291218 1.36304379 1.36209154
     1.36034524 1.3578831  1.35484791 1.35127687 1.34753394 1.3436749
     1.3398391  1.33610225 1.332551   1.32937622 1.32660317 1.32430589
     1.32252491 1.3212831  1.32058287 1.32040823 1.32072759 1.32149363
     1.32264841 1.32412374 1.32584608 1.32773757 1.32972026 1.33171952]
    46 dia. Valor Previsto -> [[1.3336644]]
    47 dia. Valores de Entrada -> [1.35923731 1.36165559 1.36291218 1.36304379 1.36209154 1.36034524
     1.3578831  1.35484791 1.35127687 1.34753394 1.3436749  1.3398391
     1.33610225 1.332551   1.32937622 1.32660317 1.32430589 1.32252491
     1.3212831  1.32058287 1.32040823 1.32072759 1.32149363 1.32264841
     1.32412374 1.32584608 1.32773757 1.32972026 1.33171952 1.33366442]
    47 dia. Valor Previsto -> [[1.3354927]]
    48 dia. Valores de Entrada -> [1.36165559 1.36291218 1.36304379 1.36209154 1.36034524 1.3578831
     1.35484791 1.35127687 1.34753394 1.3436749  1.3398391  1.33610225
     1.332551   1.32937622 1.32660317 1.32430589 1.32252491 1.3212831
     1.32058287 1.32040823 1.32072759 1.32149363 1.32264841 1.32412374
     1.32584608 1.32773757 1.32972026 1.33171952 1.33366442 1.33549273]
    48 dia. Valor Previsto -> [[1.3371508]]
    49 dia. Valores de Entrada -> [1.36291218 1.36304379 1.36209154 1.36034524 1.3578831  1.35484791
     1.35127687 1.34753394 1.3436749  1.3398391  1.33610225 1.332551
     1.32937622 1.32660317 1.32430589 1.32252491 1.3212831  1.32058287
     1.32040823 1.32072759 1.32149363 1.32264841 1.32412374 1.32584608
     1.32773757 1.32972026 1.33171952 1.33366442 1.33549273 1.33715081]
    49 dia. Valor Previsto -> [[1.3385948]]
    50 dia. Valores de Entrada -> [1.36304379 1.36209154 1.36034524 1.3578831  1.35484791 1.35127687
     1.34753394 1.3436749  1.3398391  1.33610225 1.332551   1.32937622
     1.32660317 1.32430589 1.32252491 1.3212831  1.32058287 1.32040823
     1.32072759 1.32149363 1.32264841 1.32412374 1.32584608 1.32773757
     1.32972026 1.33171952 1.33366442 1.33549273 1.33715081 1.33859479]
    50 dia. Valor Previsto -> [[1.3397919]]
    51 dia. Valores de Entrada -> [1.36209154 1.36034524 1.3578831  1.35484791 1.35127687 1.34753394
     1.3436749  1.3398391  1.33610225 1.332551   1.32937622 1.32660317
     1.32430589 1.32252491 1.3212831  1.32058287 1.32040823 1.32072759
     1.32149363 1.32264841 1.32412374 1.32584608 1.32773757 1.32972026
     1.33171952 1.33366442 1.33549273 1.33715081 1.33859479 1.33979189]
    51 dia. Valor Previsto -> [[1.3407212]]
    52 dia. Valores de Entrada -> [1.36034524 1.3578831  1.35484791 1.35127687 1.34753394 1.3436749
     1.3398391  1.33610225 1.332551   1.32937622 1.32660317 1.32430589
     1.32252491 1.3212831  1.32058287 1.32040823 1.32072759 1.32149363
     1.32264841 1.32412374 1.32584608 1.32773757 1.32972026 1.33171952
     1.33366442 1.33549273 1.33715081 1.33859479 1.33979189 1.34072125]
    52 dia. Valor Previsto -> [[1.3413726]]
    53 dia. Valores de Entrada -> [1.3578831  1.35484791 1.35127687 1.34753394 1.3436749  1.3398391
     1.33610225 1.332551   1.32937622 1.32660317 1.32430589 1.32252491
     1.3212831  1.32058287 1.32040823 1.32072759 1.32149363 1.32264841
     1.32412374 1.32584608 1.32773757 1.32972026 1.33171952 1.33366442
     1.33549273 1.33715081 1.33859479 1.33979189 1.34072125 1.34137261]
    53 dia. Valor Previsto -> [[1.341746]]
    54 dia. Valores de Entrada -> [1.35484791 1.35127687 1.34753394 1.3436749  1.3398391  1.33610225
     1.332551   1.32937622 1.32660317 1.32430589 1.32252491 1.3212831
     1.32058287 1.32040823 1.32072759 1.32149363 1.32264841 1.32412374
     1.32584608 1.32773757 1.32972026 1.33171952 1.33366442 1.33549273
     1.33715081 1.33859479 1.33979189 1.34072125 1.34137261 1.34174597]
    54 dia. Valor Previsto -> [[1.341851]]
    55 dia. Valores de Entrada -> [1.35127687 1.34753394 1.3436749  1.3398391  1.33610225 1.332551
     1.32937622 1.32660317 1.32430589 1.32252491 1.3212831  1.32058287
     1.32040823 1.32072759 1.32149363 1.32264841 1.32412374 1.32584608
     1.32773757 1.32972026 1.33171952 1.33366442 1.33549273 1.33715081
     1.33859479 1.33979189 1.34072125 1.34137261 1.34174597 1.341851  ]
    55 dia. Valor Previsto -> [[1.3417056]]
    56 dia. Valores de Entrada -> [1.34753394 1.3436749  1.3398391  1.33610225 1.332551   1.32937622
     1.32660317 1.32430589 1.32252491 1.3212831  1.32058287 1.32040823
     1.32072759 1.32149363 1.32264841 1.32412374 1.32584608 1.32773757
     1.32972026 1.33171952 1.33366442 1.33549273 1.33715081 1.33859479
     1.33979189 1.34072125 1.34137261 1.34174597 1.341851   1.34170556]
    56 dia. Valor Previsto -> [[1.3413345]]
    57 dia. Valores de Entrada -> [1.3436749  1.3398391  1.33610225 1.332551   1.32937622 1.32660317
     1.32430589 1.32252491 1.3212831  1.32058287 1.32040823 1.32072759
     1.32149363 1.32264841 1.32412374 1.32584608 1.32773757 1.32972026
     1.33171952 1.33366442 1.33549273 1.33715081 1.33859479 1.33979189
     1.34072125 1.34137261 1.34174597 1.341851   1.34170556 1.34133446]
    57 dia. Valor Previsto -> [[1.340768]]
    58 dia. Valores de Entrada -> [1.3398391  1.33610225 1.332551   1.32937622 1.32660317 1.32430589
     1.32252491 1.3212831  1.32058287 1.32040823 1.32072759 1.32149363
     1.32264841 1.32412374 1.32584608 1.32773757 1.32972026 1.33171952
     1.33366442 1.33549273 1.33715081 1.33859479 1.33979189 1.34072125
     1.34137261 1.34174597 1.341851   1.34170556 1.34133446 1.34076798]
    58 dia. Valor Previsto -> [[1.3400408]]
    59 dia. Valores de Entrada -> [1.33610225 1.332551   1.32937622 1.32660317 1.32430589 1.32252491
     1.3212831  1.32058287 1.32040823 1.32072759 1.32149363 1.32264841
     1.32412374 1.32584608 1.32773757 1.32972026 1.33171952 1.33366442
     1.33549273 1.33715081 1.33859479 1.33979189 1.34072125 1.34137261
     1.34174597 1.341851   1.34170556 1.34133446 1.34076798 1.3400408 ]
    59 dia. Valor Previsto -> [[1.3391907]]
    60 dia. Valores de Entrada -> [1.332551   1.32937622 1.32660317 1.32430589 1.32252491 1.3212831
     1.32058287 1.32040823 1.32072759 1.32149363 1.32264841 1.32412374
     1.32584608 1.32773757 1.32972026 1.33171952 1.33366442 1.33549273
     1.33715081 1.33859479 1.33979189 1.34072125 1.34137261 1.34174597
     1.341851   1.34170556 1.34133446 1.34076798 1.3400408  1.33919072]
    60 dia. Valor Previsto -> [[1.3382547]]
    61 dia. Valores de Entrada -> [1.32937622 1.32660317 1.32430589 1.32252491 1.3212831  1.32058287
     1.32040823 1.32072759 1.32149363 1.32264841 1.32412374 1.32584608
     1.32773757 1.32972026 1.33171952 1.33366442 1.33549273 1.33715081
     1.33859479 1.33979189 1.34072125 1.34137261 1.34174597 1.341851
     1.34170556 1.34133446 1.34076798 1.3400408  1.33919072 1.33825469]
    61 dia. Valor Previsto -> [[1.3372715]]
    62 dia. Valores de Entrada -> [1.32660317 1.32430589 1.32252491 1.3212831  1.32058287 1.32040823
     1.32072759 1.32149363 1.32264841 1.32412374 1.32584608 1.32773757
     1.32972026 1.33171952 1.33366442 1.33549273 1.33715081 1.33859479
     1.33979189 1.34072125 1.34137261 1.34174597 1.341851   1.34170556
     1.34133446 1.34076798 1.3400408  1.33919072 1.33825469 1.33727145]
    62 dia. Valor Previsto -> [[1.3362769]]
    63 dia. Valores de Entrada -> [1.32430589 1.32252491 1.3212831  1.32058287 1.32040823 1.32072759
     1.32149363 1.32264841 1.32412374 1.32584608 1.32773757 1.32972026
     1.33171952 1.33366442 1.33549273 1.33715081 1.33859479 1.33979189
     1.34072125 1.34137261 1.34174597 1.341851   1.34170556 1.34133446
     1.34076798 1.3400408  1.33919072 1.33825469 1.33727145 1.33627689]
    63 dia. Valor Previsto -> [[1.3353062]]
    64 dia. Valores de Entrada -> [1.32252491 1.3212831  1.32058287 1.32040823 1.32072759 1.32149363
     1.32264841 1.32412374 1.32584608 1.32773757 1.32972026 1.33171952
     1.33366442 1.33549273 1.33715081 1.33859479 1.33979189 1.34072125
     1.34137261 1.34174597 1.341851   1.34170556 1.34133446 1.34076798
     1.3400408  1.33919072 1.33825469 1.33727145 1.33627689 1.33530617]
    64 dia. Valor Previsto -> [[1.3343894]]
    65 dia. Valores de Entrada -> [1.3212831  1.32058287 1.32040823 1.32072759 1.32149363 1.32264841
     1.32412374 1.32584608 1.32773757 1.32972026 1.33171952 1.33366442
     1.33549273 1.33715081 1.33859479 1.33979189 1.34072125 1.34137261
     1.34174597 1.341851   1.34170556 1.34133446 1.34076798 1.3400408
     1.33919072 1.33825469 1.33727145 1.33627689 1.33530617 1.33438945]
    65 dia. Valor Previsto -> [[1.3335534]]
    66 dia. Valores de Entrada -> [1.32058287 1.32040823 1.32072759 1.32149363 1.32264841 1.32412374
     1.32584608 1.32773757 1.32972026 1.33171952 1.33366442 1.33549273
     1.33715081 1.33859479 1.33979189 1.34072125 1.34137261 1.34174597
     1.341851   1.34170556 1.34133446 1.34076798 1.3400408  1.33919072
     1.33825469 1.33727145 1.33627689 1.33530617 1.33438945 1.33355343]
    66 dia. Valor Previsto -> [[1.3328202]]
    67 dia. Valores de Entrada -> [1.32040823 1.32072759 1.32149363 1.32264841 1.32412374 1.32584608
     1.32773757 1.32972026 1.33171952 1.33366442 1.33549273 1.33715081
     1.33859479 1.33979189 1.34072125 1.34137261 1.34174597 1.341851
     1.34170556 1.34133446 1.34076798 1.3400408  1.33919072 1.33825469
     1.33727145 1.33627689 1.33530617 1.33438945 1.33355343 1.33282018]
    67 dia. Valor Previsto -> [[1.3322062]]
    68 dia. Valores de Entrada -> [1.32072759 1.32149363 1.32264841 1.32412374 1.32584608 1.32773757
     1.32972026 1.33171952 1.33366442 1.33549273 1.33715081 1.33859479
     1.33979189 1.34072125 1.34137261 1.34174597 1.341851   1.34170556
     1.34133446 1.34076798 1.3400408  1.33919072 1.33825469 1.33727145
     1.33627689 1.33530617 1.33438945 1.33355343 1.33282018 1.33220625]
    68 dia. Valor Previsto -> [[1.3317231]]
    69 dia. Valores de Entrada -> [1.32149363 1.32264841 1.32412374 1.32584608 1.32773757 1.32972026
     1.33171952 1.33366442 1.33549273 1.33715081 1.33859479 1.33979189
     1.34072125 1.34137261 1.34174597 1.341851   1.34170556 1.34133446
     1.34076798 1.3400408  1.33919072 1.33825469 1.33727145 1.33627689
     1.33530617 1.33438945 1.33355343 1.33282018 1.33220625 1.33172309]
    69 dia. Valor Previsto -> [[1.331377]]
    70 dia. Valores de Entrada -> [1.32264841 1.32412374 1.32584608 1.32773757 1.32972026 1.33171952
     1.33366442 1.33549273 1.33715081 1.33859479 1.33979189 1.34072125
     1.34137261 1.34174597 1.341851   1.34170556 1.34133446 1.34076798
     1.3400408  1.33919072 1.33825469 1.33727145 1.33627689 1.33530617
     1.33438945 1.33355343 1.33282018 1.33220625 1.33172309 1.33137703]
    70 dia. Valor Previsto -> [[1.3311691]]
    71 dia. Valores de Entrada -> [1.32412374 1.32584608 1.32773757 1.32972026 1.33171952 1.33366442
     1.33549273 1.33715081 1.33859479 1.33979189 1.34072125 1.34137261
     1.34174597 1.341851   1.34170556 1.34133446 1.34076798 1.3400408
     1.33919072 1.33825469 1.33727145 1.33627689 1.33530617 1.33438945
     1.33355343 1.33282018 1.33220625 1.33172309 1.33137703 1.33116913]
    71 dia. Valor Previsto -> [[1.3310965]]
    72 dia. Valores de Entrada -> [1.32584608 1.32773757 1.32972026 1.33171952 1.33366442 1.33549273
     1.33715081 1.33859479 1.33979189 1.34072125 1.34137261 1.34174597
     1.341851   1.34170556 1.34133446 1.34076798 1.3400408  1.33919072
     1.33825469 1.33727145 1.33627689 1.33530617 1.33438945 1.33355343
     1.33282018 1.33220625 1.33172309 1.33137703 1.33116913 1.33109653]
    72 dia. Valor Previsto -> [[1.3311511]]
    73 dia. Valores de Entrada -> [1.32773757 1.32972026 1.33171952 1.33366442 1.33549273 1.33715081
     1.33859479 1.33979189 1.34072125 1.34137261 1.34174597 1.341851
     1.34170556 1.34133446 1.34076798 1.3400408  1.33919072 1.33825469
     1.33727145 1.33627689 1.33530617 1.33438945 1.33355343 1.33282018
     1.33220625 1.33172309 1.33137703 1.33116913 1.33109653 1.33115113]
    73 dia. Valor Previsto -> [[1.3313216]]
    74 dia. Valores de Entrada -> [1.32972026 1.33171952 1.33366442 1.33549273 1.33715081 1.33859479
     1.33979189 1.34072125 1.34137261 1.34174597 1.341851   1.34170556
     1.34133446 1.34076798 1.3400408  1.33919072 1.33825469 1.33727145
     1.33627689 1.33530617 1.33438945 1.33355343 1.33282018 1.33220625
     1.33172309 1.33137703 1.33116913 1.33109653 1.33115113 1.3313216 ]
    74 dia. Valor Previsto -> [[1.3315928]]
    75 dia. Valores de Entrada -> [1.33171952 1.33366442 1.33549273 1.33715081 1.33859479 1.33979189
     1.34072125 1.34137261 1.34174597 1.341851   1.34170556 1.34133446
     1.34076798 1.3400408  1.33919072 1.33825469 1.33727145 1.33627689
     1.33530617 1.33438945 1.33355343 1.33282018 1.33220625 1.33172309
     1.33137703 1.33116913 1.33109653 1.33115113 1.3313216  1.3315928 ]
    75 dia. Valor Previsto -> [[1.3319485]]
    76 dia. Valores de Entrada -> [1.33366442 1.33549273 1.33715081 1.33859479 1.33979189 1.34072125
     1.34137261 1.34174597 1.341851   1.34170556 1.34133446 1.34076798
     1.3400408  1.33919072 1.33825469 1.33727145 1.33627689 1.33530617
     1.33438945 1.33355343 1.33282018 1.33220625 1.33172309 1.33137703
     1.33116913 1.33109653 1.33115113 1.3313216  1.3315928  1.33194852]
    76 dia. Valor Previsto -> [[1.33237]]
    77 dia. Valores de Entrada -> [1.33549273 1.33715081 1.33859479 1.33979189 1.34072125 1.34137261
     1.34174597 1.341851   1.34170556 1.34133446 1.34076798 1.3400408
     1.33919072 1.33825469 1.33727145 1.33627689 1.33530617 1.33438945
     1.33355343 1.33282018 1.33220625 1.33172309 1.33137703 1.33116913
     1.33109653 1.33115113 1.3313216  1.3315928  1.33194852 1.33237004]
    77 dia. Valor Previsto -> [[1.3328373]]
    78 dia. Valores de Entrada -> [1.33715081 1.33859479 1.33979189 1.34072125 1.34137261 1.34174597
     1.341851   1.34170556 1.34133446 1.34076798 1.3400408  1.33919072
     1.33825469 1.33727145 1.33627689 1.33530617 1.33438945 1.33355343
     1.33282018 1.33220625 1.33172309 1.33137703 1.33116913 1.33109653
     1.33115113 1.3313216  1.3315928  1.33194852 1.33237004 1.33283734]
    78 dia. Valor Previsto -> [[1.3333317]]
    79 dia. Valores de Entrada -> [1.33859479 1.33979189 1.34072125 1.34137261 1.34174597 1.341851
     1.34170556 1.34133446 1.34076798 1.3400408  1.33919072 1.33825469
     1.33727145 1.33627689 1.33530617 1.33438945 1.33355343 1.33282018
     1.33220625 1.33172309 1.33137703 1.33116913 1.33109653 1.33115113
     1.3313216  1.3315928  1.33194852 1.33237004 1.33283734 1.3333317 ]
    79 dia. Valor Previsto -> [[1.3338343]]
    80 dia. Valores de Entrada -> [1.33979189 1.34072125 1.34137261 1.34174597 1.341851   1.34170556
     1.34133446 1.34076798 1.3400408  1.33919072 1.33825469 1.33727145
     1.33627689 1.33530617 1.33438945 1.33355343 1.33282018 1.33220625
     1.33172309 1.33137703 1.33116913 1.33109653 1.33115113 1.3313216
     1.3315928  1.33194852 1.33237004 1.33283734 1.3333317  1.33383429]
    80 dia. Valor Previsto -> [[1.3343272]]
    81 dia. Valores de Entrada -> [1.34072125 1.34137261 1.34174597 1.341851   1.34170556 1.34133446
     1.34076798 1.3400408  1.33919072 1.33825469 1.33727145 1.33627689
     1.33530617 1.33438945 1.33355343 1.33282018 1.33220625 1.33172309
     1.33137703 1.33116913 1.33109653 1.33115113 1.3313216  1.3315928
     1.33194852 1.33237004 1.33283734 1.3333317  1.33383429 1.33432722]
    81 dia. Valor Previsto -> [[1.3347951]]
    82 dia. Valores de Entrada -> [1.34137261 1.34174597 1.341851   1.34170556 1.34133446 1.34076798
     1.3400408  1.33919072 1.33825469 1.33727145 1.33627689 1.33530617
     1.33438945 1.33355343 1.33282018 1.33220625 1.33172309 1.33137703
     1.33116913 1.33109653 1.33115113 1.3313216  1.3315928  1.33194852
     1.33237004 1.33283734 1.3333317  1.33383429 1.33432722 1.33479512]
    82 dia. Valor Previsto -> [[1.3352232]]
    83 dia. Valores de Entrada -> [1.34174597 1.341851   1.34170556 1.34133446 1.34076798 1.3400408
     1.33919072 1.33825469 1.33727145 1.33627689 1.33530617 1.33438945
     1.33355343 1.33282018 1.33220625 1.33172309 1.33137703 1.33116913
     1.33109653 1.33115113 1.3313216  1.3315928  1.33194852 1.33237004
     1.33283734 1.3333317  1.33383429 1.33432722 1.33479512 1.3352232 ]
    83 dia. Valor Previsto -> [[1.3355999]]
    84 dia. Valores de Entrada -> [1.341851   1.34170556 1.34133446 1.34076798 1.3400408  1.33919072
     1.33825469 1.33727145 1.33627689 1.33530617 1.33438945 1.33355343
     1.33282018 1.33220625 1.33172309 1.33137703 1.33116913 1.33109653
     1.33115113 1.3313216  1.3315928  1.33194852 1.33237004 1.33283734
     1.3333317  1.33383429 1.33432722 1.33479512 1.3352232  1.3355999 ]
    84 dia. Valor Previsto -> [[1.3359175]]
    85 dia. Valores de Entrada -> [1.34170556 1.34133446 1.34076798 1.3400408  1.33919072 1.33825469
     1.33727145 1.33627689 1.33530617 1.33438945 1.33355343 1.33282018
     1.33220625 1.33172309 1.33137703 1.33116913 1.33109653 1.33115113
     1.3313216  1.3315928  1.33194852 1.33237004 1.33283734 1.3333317
     1.33383429 1.33432722 1.33479512 1.3352232  1.3355999  1.33591747]
    85 dia. Valor Previsto -> [[1.336169]]
    86 dia. Valores de Entrada -> [1.34133446 1.34076798 1.3400408  1.33919072 1.33825469 1.33727145
     1.33627689 1.33530617 1.33438945 1.33355343 1.33282018 1.33220625
     1.33172309 1.33137703 1.33116913 1.33109653 1.33115113 1.3313216
     1.3315928  1.33194852 1.33237004 1.33283734 1.3333317  1.33383429
     1.33432722 1.33479512 1.3352232  1.3355999  1.33591747 1.336169  ]
    86 dia. Valor Previsto -> [[1.3363512]]
    87 dia. Valores de Entrada -> [1.34076798 1.3400408  1.33919072 1.33825469 1.33727145 1.33627689
     1.33530617 1.33438945 1.33355343 1.33282018 1.33220625 1.33172309
     1.33137703 1.33116913 1.33109653 1.33115113 1.3313216  1.3315928
     1.33194852 1.33237004 1.33283734 1.3333317  1.33383429 1.33432722
     1.33479512 1.3352232  1.3355999  1.33591747 1.336169   1.33635116]
    87 dia. Valor Previsto -> [[1.3364633]]
    88 dia. Valores de Entrada -> [1.3400408  1.33919072 1.33825469 1.33727145 1.33627689 1.33530617
     1.33438945 1.33355343 1.33282018 1.33220625 1.33172309 1.33137703
     1.33116913 1.33109653 1.33115113 1.3313216  1.3315928  1.33194852
     1.33237004 1.33283734 1.3333317  1.33383429 1.33432722 1.33479512
     1.3352232  1.3355999  1.33591747 1.336169   1.33635116 1.33646333]
    88 dia. Valor Previsto -> [[1.3365068]]
    89 dia. Valores de Entrada -> [1.33919072 1.33825469 1.33727145 1.33627689 1.33530617 1.33438945
     1.33355343 1.33282018 1.33220625 1.33172309 1.33137703 1.33116913
     1.33109653 1.33115113 1.3313216  1.3315928  1.33194852 1.33237004
     1.33283734 1.3333317  1.33383429 1.33432722 1.33479512 1.3352232
     1.3355999  1.33591747 1.336169   1.33635116 1.33646333 1.33650684]
    89 dia. Valor Previsto -> [[1.3364865]]
    90 dia. Valores de Entrada -> [1.33825469 1.33727145 1.33627689 1.33530617 1.33438945 1.33355343
     1.33282018 1.33220625 1.33172309 1.33137703 1.33116913 1.33109653
     1.33115113 1.3313216  1.3315928  1.33194852 1.33237004 1.33283734
     1.3333317  1.33383429 1.33432722 1.33479512 1.3352232  1.3355999
     1.33591747 1.336169   1.33635116 1.33646333 1.33650684 1.33648646]
    90 dia. Valor Previsto -> [[1.336407]]
    91 dia. Valores de Entrada -> [1.33727145 1.33627689 1.33530617 1.33438945 1.33355343 1.33282018
     1.33220625 1.33172309 1.33137703 1.33116913 1.33109653 1.33115113
     1.3313216  1.3315928  1.33194852 1.33237004 1.33283734 1.3333317
     1.33383429 1.33432722 1.33479512 1.3352232  1.3355999  1.33591747
     1.336169   1.33635116 1.33646333 1.33650684 1.33648646 1.33640695]
    91 dia. Valor Previsto -> [[1.3362757]]
    92 dia. Valores de Entrada -> [1.33627689 1.33530617 1.33438945 1.33355343 1.33282018 1.33220625
     1.33172309 1.33137703 1.33116913 1.33109653 1.33115113 1.3313216
     1.3315928  1.33194852 1.33237004 1.33283734 1.3333317  1.33383429
     1.33432722 1.33479512 1.3352232  1.3355999  1.33591747 1.336169
     1.33635116 1.33646333 1.33650684 1.33648646 1.33640695 1.3362757 ]
    92 dia. Valor Previsto -> [[1.3361015]]
    93 dia. Valores de Entrada -> [1.33530617 1.33438945 1.33355343 1.33282018 1.33220625 1.33172309
     1.33137703 1.33116913 1.33109653 1.33115113 1.3313216  1.3315928
     1.33194852 1.33237004 1.33283734 1.3333317  1.33383429 1.33432722
     1.33479512 1.3352232  1.3355999  1.33591747 1.336169   1.33635116
     1.33646333 1.33650684 1.33648646 1.33640695 1.3362757  1.33610153]
    93 dia. Valor Previsto -> [[1.3358933]]
    94 dia. Valores de Entrada -> [1.33438945 1.33355343 1.33282018 1.33220625 1.33172309 1.33137703
     1.33116913 1.33109653 1.33115113 1.3313216  1.3315928  1.33194852
     1.33237004 1.33283734 1.3333317  1.33383429 1.33432722 1.33479512
     1.3352232  1.3355999  1.33591747 1.336169   1.33635116 1.33646333
     1.33650684 1.33648646 1.33640695 1.3362757  1.33610153 1.33589327]
    94 dia. Valor Previsto -> [[1.3356614]]
    95 dia. Valores de Entrada -> [1.33355343 1.33282018 1.33220625 1.33172309 1.33137703 1.33116913
     1.33109653 1.33115113 1.3313216  1.3315928  1.33194852 1.33237004
     1.33283734 1.3333317  1.33383429 1.33432722 1.33479512 1.3352232
     1.3355999  1.33591747 1.336169   1.33635116 1.33646333 1.33650684
     1.33648646 1.33640695 1.3362757  1.33610153 1.33589327 1.33566141]
    95 dia. Valor Previsto -> [[1.335414]]
    96 dia. Valores de Entrada -> [1.33282018 1.33220625 1.33172309 1.33137703 1.33116913 1.33109653
     1.33115113 1.3313216  1.3315928  1.33194852 1.33237004 1.33283734
     1.3333317  1.33383429 1.33432722 1.33479512 1.3352232  1.3355999
     1.33591747 1.336169   1.33635116 1.33646333 1.33650684 1.33648646
     1.33640695 1.3362757  1.33610153 1.33589327 1.33566141 1.33541405]
    96 dia. Valor Previsto -> [[1.3351624]]
    97 dia. Valores de Entrada -> [1.33220625 1.33172309 1.33137703 1.33116913 1.33109653 1.33115113
     1.3313216  1.3315928  1.33194852 1.33237004 1.33283734 1.3333317
     1.33383429 1.33432722 1.33479512 1.3352232  1.3355999  1.33591747
     1.336169   1.33635116 1.33646333 1.33650684 1.33648646 1.33640695
     1.3362757  1.33610153 1.33589327 1.33566141 1.33541405 1.3351624 ]
    97 dia. Valor Previsto -> [[1.3349142]]
    98 dia. Valores de Entrada -> [1.33172309 1.33137703 1.33116913 1.33109653 1.33115113 1.3313216
     1.3315928  1.33194852 1.33237004 1.33283734 1.3333317  1.33383429
     1.33432722 1.33479512 1.3352232  1.3355999  1.33591747 1.336169
     1.33635116 1.33646333 1.33650684 1.33648646 1.33640695 1.3362757
     1.33610153 1.33589327 1.33566141 1.33541405 1.3351624  1.33491421]
    98 dia. Valor Previsto -> [[1.3346784]]
    99 dia. Valores de Entrada -> [1.33137703 1.33116913 1.33109653 1.33115113 1.3313216  1.3315928
     1.33194852 1.33237004 1.33283734 1.3333317  1.33383429 1.33432722
     1.33479512 1.3352232  1.3355999  1.33591747 1.336169   1.33635116
     1.33646333 1.33650684 1.33648646 1.33640695 1.3362757  1.33610153
     1.33589327 1.33566141 1.33541405 1.3351624  1.33491421 1.33467841]
    99 dia. Valor Previsto -> [[1.3344611]]
    100 dia. Valores de Entrada -> [1.33116913 1.33109653 1.33115113 1.3313216  1.3315928  1.33194852
     1.33237004 1.33283734 1.3333317  1.33383429 1.33432722 1.33479512
     1.3352232  1.3355999  1.33591747 1.336169   1.33635116 1.33646333
     1.33650684 1.33648646 1.33640695 1.3362757  1.33610153 1.33589327
     1.33566141 1.33541405 1.3351624  1.33491421 1.33467841 1.33446109]
    100 dia. Valor Previsto -> [[1.3342685]]
    101 dia. Valores de Entrada -> [1.33109653 1.33115113 1.3313216  1.3315928  1.33194852 1.33237004
     1.33283734 1.3333317  1.33383429 1.33432722 1.33479512 1.3352232
     1.3355999  1.33591747 1.336169   1.33635116 1.33646333 1.33650684
     1.33648646 1.33640695 1.3362757  1.33610153 1.33589327 1.33566141
     1.33541405 1.3351624  1.33491421 1.33467841 1.33446109 1.33426845]
    101 dia. Valor Previsto -> [[1.3341054]]
    102 dia. Valores de Entrada -> [1.33115113 1.3313216  1.3315928  1.33194852 1.33237004 1.33283734
     1.3333317  1.33383429 1.33432722 1.33479512 1.3352232  1.3355999
     1.33591747 1.336169   1.33635116 1.33646333 1.33650684 1.33648646
     1.33640695 1.3362757  1.33610153 1.33589327 1.33566141 1.33541405
     1.3351624  1.33491421 1.33467841 1.33446109 1.33426845 1.33410537]
    102 dia. Valor Previsto -> [[1.3339748]]
    103 dia. Valores de Entrada -> [1.3313216  1.3315928  1.33194852 1.33237004 1.33283734 1.3333317
     1.33383429 1.33432722 1.33479512 1.3352232  1.3355999  1.33591747
     1.336169   1.33635116 1.33646333 1.33650684 1.33648646 1.33640695
     1.3362757  1.33610153 1.33589327 1.33566141 1.33541405 1.3351624
     1.33491421 1.33467841 1.33446109 1.33426845 1.33410537 1.33397484]
    103 dia. Valor Previsto -> [[1.3338791]]
    104 dia. Valores de Entrada -> [1.3315928  1.33194852 1.33237004 1.33283734 1.3333317  1.33383429
     1.33432722 1.33479512 1.3352232  1.3355999  1.33591747 1.336169
     1.33635116 1.33646333 1.33650684 1.33648646 1.33640695 1.3362757
     1.33610153 1.33589327 1.33566141 1.33541405 1.3351624  1.33491421
     1.33467841 1.33446109 1.33426845 1.33410537 1.33397484 1.33387911]
    104 dia. Valor Previsto -> [[1.3338181]]
    105 dia. Valores de Entrada -> [1.33194852 1.33237004 1.33283734 1.3333317  1.33383429 1.33432722
     1.33479512 1.3352232  1.3355999  1.33591747 1.336169   1.33635116
     1.33646333 1.33650684 1.33648646 1.33640695 1.3362757  1.33610153
     1.33589327 1.33566141 1.33541405 1.3351624  1.33491421 1.33467841
     1.33446109 1.33426845 1.33410537 1.33397484 1.33387911 1.33381808]
    105 dia. Valor Previsto -> [[1.333792]]
    106 dia. Valores de Entrada -> [1.33237004 1.33283734 1.3333317  1.33383429 1.33432722 1.33479512
     1.3352232  1.3355999  1.33591747 1.336169   1.33635116 1.33646333
     1.33650684 1.33648646 1.33640695 1.3362757  1.33610153 1.33589327
     1.33566141 1.33541405 1.3351624  1.33491421 1.33467841 1.33446109
     1.33426845 1.33410537 1.33397484 1.33387911 1.33381808 1.33379197]
    106 dia. Valor Previsto -> [[1.3337983]]
    107 dia. Valores de Entrada -> [1.33283734 1.3333317  1.33383429 1.33432722 1.33479512 1.3352232
     1.3355999  1.33591747 1.336169   1.33635116 1.33646333 1.33650684
     1.33648646 1.33640695 1.3362757  1.33610153 1.33589327 1.33566141
     1.33541405 1.3351624  1.33491421 1.33467841 1.33446109 1.33426845
     1.33410537 1.33397484 1.33387911 1.33381808 1.33379197 1.33379829]
    107 dia. Valor Previsto -> [[1.3338345]]
    108 dia. Valores de Entrada -> [1.3333317  1.33383429 1.33432722 1.33479512 1.3352232  1.3355999
     1.33591747 1.336169   1.33635116 1.33646333 1.33650684 1.33648646
     1.33640695 1.3362757  1.33610153 1.33589327 1.33566141 1.33541405
     1.3351624  1.33491421 1.33467841 1.33446109 1.33426845 1.33410537
     1.33397484 1.33387911 1.33381808 1.33379197 1.33379829 1.33383453]
    108 dia. Valor Previsto -> [[1.3338971]]
    109 dia. Valores de Entrada -> [1.33383429 1.33432722 1.33479512 1.3352232  1.3355999  1.33591747
     1.336169   1.33635116 1.33646333 1.33650684 1.33648646 1.33640695
     1.3362757  1.33610153 1.33589327 1.33566141 1.33541405 1.3351624
     1.33491421 1.33467841 1.33446109 1.33426845 1.33410537 1.33397484
     1.33387911 1.33381808 1.33379197 1.33379829 1.33383453 1.33389711]
    109 dia. Valor Previsto -> [[1.3339822]]
    110 dia. Valores de Entrada -> [1.33432722 1.33479512 1.3352232  1.3355999  1.33591747 1.336169
     1.33635116 1.33646333 1.33650684 1.33648646 1.33640695 1.3362757
     1.33610153 1.33589327 1.33566141 1.33541405 1.3351624  1.33491421
     1.33467841 1.33446109 1.33426845 1.33410537 1.33397484 1.33387911
     1.33381808 1.33379197 1.33379829 1.33383453 1.33389711 1.33398223]
    110 dia. Valor Previsto -> [[1.3340845]]
    111 dia. Valores de Entrada -> [1.33479512 1.3352232  1.3355999  1.33591747 1.336169   1.33635116
     1.33646333 1.33650684 1.33648646 1.33640695 1.3362757  1.33610153
     1.33589327 1.33566141 1.33541405 1.3351624  1.33491421 1.33467841
     1.33446109 1.33426845 1.33410537 1.33397484 1.33387911 1.33381808
     1.33379197 1.33379829 1.33383453 1.33389711 1.33398223 1.33408451]
    111 dia. Valor Previsto -> [[1.3342004]]
    112 dia. Valores de Entrada -> [1.3352232  1.3355999  1.33591747 1.336169   1.33635116 1.33646333
     1.33650684 1.33648646 1.33640695 1.3362757  1.33610153 1.33589327
     1.33566141 1.33541405 1.3351624  1.33491421 1.33467841 1.33446109
     1.33426845 1.33410537 1.33397484 1.33387911 1.33381808 1.33379197
     1.33379829 1.33383453 1.33389711 1.33398223 1.33408451 1.33420038]
    112 dia. Valor Previsto -> [[1.3343238]]
    113 dia. Valores de Entrada -> [1.3355999  1.33591747 1.336169   1.33635116 1.33646333 1.33650684
     1.33648646 1.33640695 1.3362757  1.33610153 1.33589327 1.33566141
     1.33541405 1.3351624  1.33491421 1.33467841 1.33446109 1.33426845
     1.33410537 1.33397484 1.33387911 1.33381808 1.33379197 1.33379829
     1.33383453 1.33389711 1.33398223 1.33408451 1.33420038 1.33432376]
    113 dia. Valor Previsto -> [[1.3344505]]
    114 dia. Valores de Entrada -> [1.33591747 1.336169   1.33635116 1.33646333 1.33650684 1.33648646
     1.33640695 1.3362757  1.33610153 1.33589327 1.33566141 1.33541405
     1.3351624  1.33491421 1.33467841 1.33446109 1.33426845 1.33410537
     1.33397484 1.33387911 1.33381808 1.33379197 1.33379829 1.33383453
     1.33389711 1.33398223 1.33408451 1.33420038 1.33432376 1.33445048]
    114 dia. Valor Previsto -> [[1.334576]]
    115 dia. Valores de Entrada -> [1.336169   1.33635116 1.33646333 1.33650684 1.33648646 1.33640695
     1.3362757  1.33610153 1.33589327 1.33566141 1.33541405 1.3351624
     1.33491421 1.33467841 1.33446109 1.33426845 1.33410537 1.33397484
     1.33387911 1.33381808 1.33379197 1.33379829 1.33383453 1.33389711
     1.33398223 1.33408451 1.33420038 1.33432376 1.33445048 1.33457601]
    115 dia. Valor Previsto -> [[1.3346957]]
    116 dia. Valores de Entrada -> [1.33635116 1.33646333 1.33650684 1.33648646 1.33640695 1.3362757
     1.33610153 1.33589327 1.33566141 1.33541405 1.3351624  1.33491421
     1.33467841 1.33446109 1.33426845 1.33410537 1.33397484 1.33387911
     1.33381808 1.33379197 1.33379829 1.33383453 1.33389711 1.33398223
     1.33408451 1.33420038 1.33432376 1.33445048 1.33457601 1.3346957 ]
    116 dia. Valor Previsto -> [[1.3348063]]
    117 dia. Valores de Entrada -> [1.33646333 1.33650684 1.33648646 1.33640695 1.3362757  1.33610153
     1.33589327 1.33566141 1.33541405 1.3351624  1.33491421 1.33467841
     1.33446109 1.33426845 1.33410537 1.33397484 1.33387911 1.33381808
     1.33379197 1.33379829 1.33383453 1.33389711 1.33398223 1.33408451
     1.33420038 1.33432376 1.33445048 1.33457601 1.3346957  1.33480632]
    117 dia. Valor Previsto -> [[1.3349053]]
    118 dia. Valores de Entrada -> [1.33650684 1.33648646 1.33640695 1.3362757  1.33610153 1.33589327
     1.33566141 1.33541405 1.3351624  1.33491421 1.33467841 1.33446109
     1.33426845 1.33410537 1.33397484 1.33387911 1.33381808 1.33379197
     1.33379829 1.33383453 1.33389711 1.33398223 1.33408451 1.33420038
     1.33432376 1.33445048 1.33457601 1.3346957  1.33480632 1.33490527]
    118 dia. Valor Previsto -> [[1.3349891]]
    119 dia. Valores de Entrada -> [1.33648646 1.33640695 1.3362757  1.33610153 1.33589327 1.33566141
     1.33541405 1.3351624  1.33491421 1.33467841 1.33446109 1.33426845
     1.33410537 1.33397484 1.33387911 1.33381808 1.33379197 1.33379829
     1.33383453 1.33389711 1.33398223 1.33408451 1.33420038 1.33432376
     1.33445048 1.33457601 1.3346957  1.33480632 1.33490527 1.33498907]
    119 dia. Valor Previsto -> [[1.3350565]]
    120 dia. Valores de Entrada -> [1.33640695 1.3362757  1.33610153 1.33589327 1.33566141 1.33541405
     1.3351624  1.33491421 1.33467841 1.33446109 1.33426845 1.33410537
     1.33397484 1.33387911 1.33381808 1.33379197 1.33379829 1.33383453
     1.33389711 1.33398223 1.33408451 1.33420038 1.33432376 1.33445048
     1.33457601 1.3346957  1.33480632 1.33490527 1.33498907 1.33505654]
    120 dia. Valor Previsto -> [[1.3351071]]
    121 dia. Valores de Entrada -> [1.3362757  1.33610153 1.33589327 1.33566141 1.33541405 1.3351624
     1.33491421 1.33467841 1.33446109 1.33426845 1.33410537 1.33397484
     1.33387911 1.33381808 1.33379197 1.33379829 1.33383453 1.33389711
     1.33398223 1.33408451 1.33420038 1.33432376 1.33445048 1.33457601
     1.3346957  1.33480632 1.33490527 1.33498907 1.33505654 1.33510709]
    121 dia. Valor Previsto -> [[1.3351396]]
    122 dia. Valores de Entrada -> [1.33610153 1.33589327 1.33566141 1.33541405 1.3351624  1.33491421
     1.33467841 1.33446109 1.33426845 1.33410537 1.33397484 1.33387911
     1.33381808 1.33379197 1.33379829 1.33383453 1.33389711 1.33398223
     1.33408451 1.33420038 1.33432376 1.33445048 1.33457601 1.3346957
     1.33480632 1.33490527 1.33498907 1.33505654 1.33510709 1.33513963]
    122 dia. Valor Previsto -> [[1.3351552]]
    123 dia. Valores de Entrada -> [1.33589327 1.33566141 1.33541405 1.3351624  1.33491421 1.33467841
     1.33446109 1.33426845 1.33410537 1.33397484 1.33387911 1.33381808
     1.33379197 1.33379829 1.33383453 1.33389711 1.33398223 1.33408451
     1.33420038 1.33432376 1.33445048 1.33457601 1.3346957  1.33480632
     1.33490527 1.33498907 1.33505654 1.33510709 1.33513963 1.33515525]
    123 dia. Valor Previsto -> [[1.3351538]]
    124 dia. Valores de Entrada -> [1.33566141 1.33541405 1.3351624  1.33491421 1.33467841 1.33446109
     1.33426845 1.33410537 1.33397484 1.33387911 1.33381808 1.33379197
     1.33379829 1.33383453 1.33389711 1.33398223 1.33408451 1.33420038
     1.33432376 1.33445048 1.33457601 1.3346957  1.33480632 1.33490527
     1.33498907 1.33505654 1.33510709 1.33513963 1.33515525 1.33515382]
    124 dia. Valor Previsto -> [[1.3351374]]
    125 dia. Valores de Entrada -> [1.33541405 1.3351624  1.33491421 1.33467841 1.33446109 1.33426845
     1.33410537 1.33397484 1.33387911 1.33381808 1.33379197 1.33379829
     1.33383453 1.33389711 1.33398223 1.33408451 1.33420038 1.33432376
     1.33445048 1.33457601 1.3346957  1.33480632 1.33490527 1.33498907
     1.33505654 1.33510709 1.33513963 1.33515525 1.33515382 1.33513737]
    125 dia. Valor Previsto -> [[1.3351074]]
    126 dia. Valores de Entrada -> [1.3351624  1.33491421 1.33467841 1.33446109 1.33426845 1.33410537
     1.33397484 1.33387911 1.33381808 1.33379197 1.33379829 1.33383453
     1.33389711 1.33398223 1.33408451 1.33420038 1.33432376 1.33445048
     1.33457601 1.3346957  1.33480632 1.33490527 1.33498907 1.33505654
     1.33510709 1.33513963 1.33515525 1.33515382 1.33513737 1.33510745]
    126 dia. Valor Previsto -> [[1.3350658]]
    127 dia. Valores de Entrada -> [1.33491421 1.33467841 1.33446109 1.33426845 1.33410537 1.33397484
     1.33387911 1.33381808 1.33379197 1.33379829 1.33383453 1.33389711
     1.33398223 1.33408451 1.33420038 1.33432376 1.33445048 1.33457601
     1.3346957  1.33480632 1.33490527 1.33498907 1.33505654 1.33510709
     1.33513963 1.33515525 1.33515382 1.33513737 1.33510745 1.33506584]
    127 dia. Valor Previsto -> [[1.3350153]]
    128 dia. Valores de Entrada -> [1.33467841 1.33446109 1.33426845 1.33410537 1.33397484 1.33387911
     1.33381808 1.33379197 1.33379829 1.33383453 1.33389711 1.33398223
     1.33408451 1.33420038 1.33432376 1.33445048 1.33457601 1.3346957
     1.33480632 1.33490527 1.33498907 1.33505654 1.33510709 1.33513963
     1.33515525 1.33515382 1.33513737 1.33510745 1.33506584 1.3350153 ]
    128 dia. Valor Previsto -> [[1.3349578]]
    129 dia. Valores de Entrada -> [1.33446109 1.33426845 1.33410537 1.33397484 1.33387911 1.33381808
     1.33379197 1.33379829 1.33383453 1.33389711 1.33398223 1.33408451
     1.33420038 1.33432376 1.33445048 1.33457601 1.3346957  1.33480632
     1.33490527 1.33498907 1.33505654 1.33510709 1.33513963 1.33515525
     1.33515382 1.33513737 1.33510745 1.33506584 1.3350153  1.33495784]
    129 dia. Valor Previsto -> [[1.3348962]]
    130 dia. Valores de Entrada -> [1.33426845 1.33410537 1.33397484 1.33387911 1.33381808 1.33379197
     1.33379829 1.33383453 1.33389711 1.33398223 1.33408451 1.33420038
     1.33432376 1.33445048 1.33457601 1.3346957  1.33480632 1.33490527
     1.33498907 1.33505654 1.33510709 1.33513963 1.33515525 1.33515382
     1.33513737 1.33510745 1.33506584 1.3350153  1.33495784 1.33489621]
    130 dia. Valor Previsto -> [[1.3348328]]
    131 dia. Valores de Entrada -> [1.33410537 1.33397484 1.33387911 1.33381808 1.33379197 1.33379829
     1.33383453 1.33389711 1.33398223 1.33408451 1.33420038 1.33432376
     1.33445048 1.33457601 1.3346957  1.33480632 1.33490527 1.33498907
     1.33505654 1.33510709 1.33513963 1.33515525 1.33515382 1.33513737
     1.33510745 1.33506584 1.3350153  1.33495784 1.33489621 1.33483279]
    131 dia. Valor Previsto -> [[1.3347695]]
    132 dia. Valores de Entrada -> [1.33397484 1.33387911 1.33381808 1.33379197 1.33379829 1.33383453
     1.33389711 1.33398223 1.33408451 1.33420038 1.33432376 1.33445048
     1.33457601 1.3346957  1.33480632 1.33490527 1.33498907 1.33505654
     1.33510709 1.33513963 1.33515525 1.33515382 1.33513737 1.33510745
     1.33506584 1.3350153  1.33495784 1.33489621 1.33483279 1.33476949]
    132 dia. Valor Previsto -> [[1.3347088]]
    133 dia. Valores de Entrada -> [1.33387911 1.33381808 1.33379197 1.33379829 1.33383453 1.33389711
     1.33398223 1.33408451 1.33420038 1.33432376 1.33445048 1.33457601
     1.3346957  1.33480632 1.33490527 1.33498907 1.33505654 1.33510709
     1.33513963 1.33515525 1.33515382 1.33513737 1.33510745 1.33506584
     1.3350153  1.33495784 1.33489621 1.33483279 1.33476949 1.33470881]
    133 dia. Valor Previsto -> [[1.3346525]]
    134 dia. Valores de Entrada -> [1.33381808 1.33379197 1.33379829 1.33383453 1.33389711 1.33398223
     1.33408451 1.33420038 1.33432376 1.33445048 1.33457601 1.3346957
     1.33480632 1.33490527 1.33498907 1.33505654 1.33510709 1.33513963
     1.33515525 1.33515382 1.33513737 1.33510745 1.33506584 1.3350153
     1.33495784 1.33489621 1.33483279 1.33476949 1.33470881 1.33465254]
    134 dia. Valor Previsto -> [[1.3346025]]
    135 dia. Valores de Entrada -> [1.33379197 1.33379829 1.33383453 1.33389711 1.33398223 1.33408451
     1.33420038 1.33432376 1.33445048 1.33457601 1.3346957  1.33480632
     1.33490527 1.33498907 1.33505654 1.33510709 1.33513963 1.33515525
     1.33515382 1.33513737 1.33510745 1.33506584 1.3350153  1.33495784
     1.33489621 1.33483279 1.33476949 1.33470881 1.33465254 1.33460248]
    135 dia. Valor Previsto -> [[1.3345593]]
    136 dia. Valores de Entrada -> [1.33379829 1.33383453 1.33389711 1.33398223 1.33408451 1.33420038
     1.33432376 1.33445048 1.33457601 1.3346957  1.33480632 1.33490527
     1.33498907 1.33505654 1.33510709 1.33513963 1.33515525 1.33515382
     1.33513737 1.33510745 1.33506584 1.3350153  1.33495784 1.33489621
     1.33483279 1.33476949 1.33470881 1.33465254 1.33460248 1.33455932]
    136 dia. Valor Previsto -> [[1.3345242]]
    137 dia. Valores de Entrada -> [1.33383453 1.33389711 1.33398223 1.33408451 1.33420038 1.33432376
     1.33445048 1.33457601 1.3346957  1.33480632 1.33490527 1.33498907
     1.33505654 1.33510709 1.33513963 1.33515525 1.33515382 1.33513737
     1.33510745 1.33506584 1.3350153  1.33495784 1.33489621 1.33483279
     1.33476949 1.33470881 1.33465254 1.33460248 1.33455932 1.33452415]
    137 dia. Valor Previsto -> [[1.3344976]]
    138 dia. Valores de Entrada -> [1.33389711 1.33398223 1.33408451 1.33420038 1.33432376 1.33445048
     1.33457601 1.3346957  1.33480632 1.33490527 1.33498907 1.33505654
     1.33510709 1.33513963 1.33515525 1.33515382 1.33513737 1.33510745
     1.33506584 1.3350153  1.33495784 1.33489621 1.33483279 1.33476949
     1.33470881 1.33465254 1.33460248 1.33455932 1.33452415 1.33449757]
    138 dia. Valor Previsto -> [[1.3344803]]
    139 dia. Valores de Entrada -> [1.33398223 1.33408451 1.33420038 1.33432376 1.33445048 1.33457601
     1.3346957  1.33480632 1.33490527 1.33498907 1.33505654 1.33510709
     1.33513963 1.33515525 1.33515382 1.33513737 1.33510745 1.33506584
     1.3350153  1.33495784 1.33489621 1.33483279 1.33476949 1.33470881
     1.33465254 1.33460248 1.33455932 1.33452415 1.33449757 1.33448029]
    139 dia. Valor Previsto -> [[1.3344716]]
    140 dia. Valores de Entrada -> [1.33408451 1.33420038 1.33432376 1.33445048 1.33457601 1.3346957
     1.33480632 1.33490527 1.33498907 1.33505654 1.33510709 1.33513963
     1.33515525 1.33515382 1.33513737 1.33510745 1.33506584 1.3350153
     1.33495784 1.33489621 1.33483279 1.33476949 1.33470881 1.33465254
     1.33460248 1.33455932 1.33452415 1.33449757 1.33448029 1.33447158]
    140 dia. Valor Previsto -> [[1.334471]]
    141 dia. Valores de Entrada -> [1.33420038 1.33432376 1.33445048 1.33457601 1.3346957  1.33480632
     1.33490527 1.33498907 1.33505654 1.33510709 1.33513963 1.33515525
     1.33515382 1.33513737 1.33510745 1.33506584 1.3350153  1.33495784
     1.33489621 1.33483279 1.33476949 1.33470881 1.33465254 1.33460248
     1.33455932 1.33452415 1.33449757 1.33448029 1.33447158 1.33447099]
    141 dia. Valor Previsto -> [[1.3344783]]
    142 dia. Valores de Entrada -> [1.33432376 1.33445048 1.33457601 1.3346957  1.33480632 1.33490527
     1.33498907 1.33505654 1.33510709 1.33513963 1.33515525 1.33515382
     1.33513737 1.33510745 1.33506584 1.3350153  1.33495784 1.33489621
     1.33483279 1.33476949 1.33470881 1.33465254 1.33460248 1.33455932
     1.33452415 1.33449757 1.33448029 1.33447158 1.33447099 1.33447826]
    142 dia. Valor Previsto -> [[1.3344926]]
    143 dia. Valores de Entrada -> [1.33445048 1.33457601 1.3346957  1.33480632 1.33490527 1.33498907
     1.33505654 1.33510709 1.33513963 1.33515525 1.33515382 1.33513737
     1.33510745 1.33506584 1.3350153  1.33495784 1.33489621 1.33483279
     1.33476949 1.33470881 1.33465254 1.33460248 1.33455932 1.33452415
     1.33449757 1.33448029 1.33447158 1.33447099 1.33447826 1.33449256]
    143 dia. Valor Previsto -> [[1.3345126]]
    144 dia. Valores de Entrada -> [1.33457601 1.3346957  1.33480632 1.33490527 1.33498907 1.33505654
     1.33510709 1.33513963 1.33515525 1.33515382 1.33513737 1.33510745
     1.33506584 1.3350153  1.33495784 1.33489621 1.33483279 1.33476949
     1.33470881 1.33465254 1.33460248 1.33455932 1.33452415 1.33449757
     1.33448029 1.33447158 1.33447099 1.33447826 1.33449256 1.33451259]
    144 dia. Valor Previsto -> [[1.3345379]]
    145 dia. Valores de Entrada -> [1.3346957  1.33480632 1.33490527 1.33498907 1.33505654 1.33510709
     1.33513963 1.33515525 1.33515382 1.33513737 1.33510745 1.33506584
     1.3350153  1.33495784 1.33489621 1.33483279 1.33476949 1.33470881
     1.33465254 1.33460248 1.33455932 1.33452415 1.33449757 1.33448029
     1.33447158 1.33447099 1.33447826 1.33449256 1.33451259 1.33453786]
    145 dia. Valor Previsto -> [[1.3345661]]
    146 dia. Valores de Entrada -> [1.33480632 1.33490527 1.33498907 1.33505654 1.33510709 1.33513963
     1.33515525 1.33515382 1.33513737 1.33510745 1.33506584 1.3350153
     1.33495784 1.33489621 1.33483279 1.33476949 1.33470881 1.33465254
     1.33460248 1.33455932 1.33452415 1.33449757 1.33448029 1.33447158
     1.33447099 1.33447826 1.33449256 1.33451259 1.33453786 1.33456612]
    146 dia. Valor Previsto -> [[1.3345971]]
    147 dia. Valores de Entrada -> [1.33490527 1.33498907 1.33505654 1.33510709 1.33513963 1.33515525
     1.33515382 1.33513737 1.33510745 1.33506584 1.3350153  1.33495784
     1.33489621 1.33483279 1.33476949 1.33470881 1.33465254 1.33460248
     1.33455932 1.33452415 1.33449757 1.33448029 1.33447158 1.33447099
     1.33447826 1.33449256 1.33451259 1.33453786 1.33456612 1.33459711]
    147 dia. Valor Previsto -> [[1.334629]]
    148 dia. Valores de Entrada -> [1.33498907 1.33505654 1.33510709 1.33513963 1.33515525 1.33515382
     1.33513737 1.33510745 1.33506584 1.3350153  1.33495784 1.33489621
     1.33483279 1.33476949 1.33470881 1.33465254 1.33460248 1.33455932
     1.33452415 1.33449757 1.33448029 1.33447158 1.33447099 1.33447826
     1.33449256 1.33451259 1.33453786 1.33456612 1.33459711 1.33462906]
    148 dia. Valor Previsto -> [[1.3346609]]
    149 dia. Valores de Entrada -> [1.33505654 1.33510709 1.33513963 1.33515525 1.33515382 1.33513737
     1.33510745 1.33506584 1.3350153  1.33495784 1.33489621 1.33483279
     1.33476949 1.33470881 1.33465254 1.33460248 1.33455932 1.33452415
     1.33449757 1.33448029 1.33447158 1.33447099 1.33447826 1.33449256
     1.33451259 1.33453786 1.33456612 1.33459711 1.33462906 1.33466089]
    149 dia. Valor Previsto -> [[1.3346912]]
    150 dia. Valores de Entrada -> [1.33510709 1.33513963 1.33515525 1.33515382 1.33513737 1.33510745
     1.33506584 1.3350153  1.33495784 1.33489621 1.33483279 1.33476949
     1.33470881 1.33465254 1.33460248 1.33455932 1.33452415 1.33449757
     1.33448029 1.33447158 1.33447099 1.33447826 1.33449256 1.33451259
     1.33453786 1.33456612 1.33459711 1.33462906 1.33466089 1.33469117]
    150 dia. Valor Previsto -> [[1.3347198]]
    151 dia. Valores de Entrada -> [1.33513963 1.33515525 1.33515382 1.33513737 1.33510745 1.33506584
     1.3350153  1.33495784 1.33489621 1.33483279 1.33476949 1.33470881
     1.33465254 1.33460248 1.33455932 1.33452415 1.33449757 1.33448029
     1.33447158 1.33447099 1.33447826 1.33449256 1.33451259 1.33453786
     1.33456612 1.33459711 1.33462906 1.33466089 1.33469117 1.33471978]
    151 dia. Valor Previsto -> [[1.3347456]]
    152 dia. Valores de Entrada -> [1.33515525 1.33515382 1.33513737 1.33510745 1.33506584 1.3350153
     1.33495784 1.33489621 1.33483279 1.33476949 1.33470881 1.33465254
     1.33460248 1.33455932 1.33452415 1.33449757 1.33448029 1.33447158
     1.33447099 1.33447826 1.33449256 1.33451259 1.33453786 1.33456612
     1.33459711 1.33462906 1.33466089 1.33469117 1.33471978 1.33474565]
    152 dia. Valor Previsto -> [[1.3347676]]
    153 dia. Valores de Entrada -> [1.33515382 1.33513737 1.33510745 1.33506584 1.3350153  1.33495784
     1.33489621 1.33483279 1.33476949 1.33470881 1.33465254 1.33460248
     1.33455932 1.33452415 1.33449757 1.33448029 1.33447158 1.33447099
     1.33447826 1.33449256 1.33451259 1.33453786 1.33456612 1.33459711
     1.33462906 1.33466089 1.33469117 1.33471978 1.33474565 1.33476758]
    153 dia. Valor Previsto -> [[1.3347858]]
    154 dia. Valores de Entrada -> [1.33513737 1.33510745 1.33506584 1.3350153  1.33495784 1.33489621
     1.33483279 1.33476949 1.33470881 1.33465254 1.33460248 1.33455932
     1.33452415 1.33449757 1.33448029 1.33447158 1.33447099 1.33447826
     1.33449256 1.33451259 1.33453786 1.33456612 1.33459711 1.33462906
     1.33466089 1.33469117 1.33471978 1.33474565 1.33476758 1.33478582]
    154 dia. Valor Previsto -> [[1.3347999]]
    155 dia. Valores de Entrada -> [1.33510745 1.33506584 1.3350153  1.33495784 1.33489621 1.33483279
     1.33476949 1.33470881 1.33465254 1.33460248 1.33455932 1.33452415
     1.33449757 1.33448029 1.33447158 1.33447099 1.33447826 1.33449256
     1.33451259 1.33453786 1.33456612 1.33459711 1.33462906 1.33466089
     1.33469117 1.33471978 1.33474565 1.33476758 1.33478582 1.33479989]
    155 dia. Valor Previsto -> [[1.3348086]]
    156 dia. Valores de Entrada -> [1.33506584 1.3350153  1.33495784 1.33489621 1.33483279 1.33476949
     1.33470881 1.33465254 1.33460248 1.33455932 1.33452415 1.33449757
     1.33448029 1.33447158 1.33447099 1.33447826 1.33449256 1.33451259
     1.33453786 1.33456612 1.33459711 1.33462906 1.33466089 1.33469117
     1.33471978 1.33474565 1.33476758 1.33478582 1.33479989 1.33480859]
    156 dia. Valor Previsto -> [[1.3348138]]
    157 dia. Valores de Entrada -> [1.3350153  1.33495784 1.33489621 1.33483279 1.33476949 1.33470881
     1.33465254 1.33460248 1.33455932 1.33452415 1.33449757 1.33448029
     1.33447158 1.33447099 1.33447826 1.33449256 1.33451259 1.33453786
     1.33456612 1.33459711 1.33462906 1.33466089 1.33469117 1.33471978
     1.33474565 1.33476758 1.33478582 1.33479989 1.33480859 1.33481383]
    157 dia. Valor Previsto -> [[1.3348143]]
    158 dia. Valores de Entrada -> [1.33495784 1.33489621 1.33483279 1.33476949 1.33470881 1.33465254
     1.33460248 1.33455932 1.33452415 1.33449757 1.33448029 1.33447158
     1.33447099 1.33447826 1.33449256 1.33451259 1.33453786 1.33456612
     1.33459711 1.33462906 1.33466089 1.33469117 1.33471978 1.33474565
     1.33476758 1.33478582 1.33479989 1.33480859 1.33481383 1.33481431]
    158 dia. Valor Previsto -> [[1.3348113]]
    159 dia. Valores de Entrada -> [1.33489621 1.33483279 1.33476949 1.33470881 1.33465254 1.33460248
     1.33455932 1.33452415 1.33449757 1.33448029 1.33447158 1.33447099
     1.33447826 1.33449256 1.33451259 1.33453786 1.33456612 1.33459711
     1.33462906 1.33466089 1.33469117 1.33471978 1.33474565 1.33476758
     1.33478582 1.33479989 1.33480859 1.33481383 1.33481431 1.33481133]
    159 dia. Valor Previsto -> [[1.3348045]]
    160 dia. Valores de Entrada -> [1.33483279 1.33476949 1.33470881 1.33465254 1.33460248 1.33455932
     1.33452415 1.33449757 1.33448029 1.33447158 1.33447099 1.33447826
     1.33449256 1.33451259 1.33453786 1.33456612 1.33459711 1.33462906
     1.33466089 1.33469117 1.33471978 1.33474565 1.33476758 1.33478582
     1.33479989 1.33480859 1.33481383 1.33481431 1.33481133 1.33480453]
    160 dia. Valor Previsto -> [[1.3347948]]
    161 dia. Valores de Entrada -> [1.33476949 1.33470881 1.33465254 1.33460248 1.33455932 1.33452415
     1.33449757 1.33448029 1.33447158 1.33447099 1.33447826 1.33449256
     1.33451259 1.33453786 1.33456612 1.33459711 1.33462906 1.33466089
     1.33469117 1.33471978 1.33474565 1.33476758 1.33478582 1.33479989
     1.33480859 1.33481383 1.33481431 1.33481133 1.33480453 1.33479476]
    161 dia. Valor Previsto -> [[1.3347828]]
    162 dia. Valores de Entrada -> [1.33470881 1.33465254 1.33460248 1.33455932 1.33452415 1.33449757
     1.33448029 1.33447158 1.33447099 1.33447826 1.33449256 1.33451259
     1.33453786 1.33456612 1.33459711 1.33462906 1.33466089 1.33469117
     1.33471978 1.33474565 1.33476758 1.33478582 1.33479989 1.33480859
     1.33481383 1.33481431 1.33481133 1.33480453 1.33479476 1.33478284]
    162 dia. Valor Previsto -> [[1.3347685]]
    163 dia. Valores de Entrada -> [1.33465254 1.33460248 1.33455932 1.33452415 1.33449757 1.33448029
     1.33447158 1.33447099 1.33447826 1.33449256 1.33451259 1.33453786
     1.33456612 1.33459711 1.33462906 1.33466089 1.33469117 1.33471978
     1.33474565 1.33476758 1.33478582 1.33479989 1.33480859 1.33481383
     1.33481431 1.33481133 1.33480453 1.33479476 1.33478284 1.33476853]
    163 dia. Valor Previsto -> [[1.334753]]
    164 dia. Valores de Entrada -> [1.33460248 1.33455932 1.33452415 1.33449757 1.33448029 1.33447158
     1.33447099 1.33447826 1.33449256 1.33451259 1.33453786 1.33456612
     1.33459711 1.33462906 1.33466089 1.33469117 1.33471978 1.33474565
     1.33476758 1.33478582 1.33479989 1.33480859 1.33481383 1.33481431
     1.33481133 1.33480453 1.33479476 1.33478284 1.33476853 1.33475304]
    164 dia. Valor Previsto -> [[1.3347372]]
    165 dia. Valores de Entrada -> [1.33455932 1.33452415 1.33449757 1.33448029 1.33447158 1.33447099
     1.33447826 1.33449256 1.33451259 1.33453786 1.33456612 1.33459711
     1.33462906 1.33466089 1.33469117 1.33471978 1.33474565 1.33476758
     1.33478582 1.33479989 1.33480859 1.33481383 1.33481431 1.33481133
     1.33480453 1.33479476 1.33478284 1.33476853 1.33475304 1.33473718]
    165 dia. Valor Previsto -> [[1.3347211]]
    166 dia. Valores de Entrada -> [1.33452415 1.33449757 1.33448029 1.33447158 1.33447099 1.33447826
     1.33449256 1.33451259 1.33453786 1.33456612 1.33459711 1.33462906
     1.33466089 1.33469117 1.33471978 1.33474565 1.33476758 1.33478582
     1.33479989 1.33480859 1.33481383 1.33481431 1.33481133 1.33480453
     1.33479476 1.33478284 1.33476853 1.33475304 1.33473718 1.33472109]
    166 dia. Valor Previsto -> [[1.3347055]]
    167 dia. Valores de Entrada -> [1.33449757 1.33448029 1.33447158 1.33447099 1.33447826 1.33449256
     1.33451259 1.33453786 1.33456612 1.33459711 1.33462906 1.33466089
     1.33469117 1.33471978 1.33474565 1.33476758 1.33478582 1.33479989
     1.33480859 1.33481383 1.33481431 1.33481133 1.33480453 1.33479476
     1.33478284 1.33476853 1.33475304 1.33473718 1.33472109 1.33470547]
    167 dia. Valor Previsto -> [[1.3346913]]
    168 dia. Valores de Entrada -> [1.33448029 1.33447158 1.33447099 1.33447826 1.33449256 1.33451259
     1.33453786 1.33456612 1.33459711 1.33462906 1.33466089 1.33469117
     1.33471978 1.33474565 1.33476758 1.33478582 1.33479989 1.33480859
     1.33481383 1.33481431 1.33481133 1.33480453 1.33479476 1.33478284
     1.33476853 1.33475304 1.33473718 1.33472109 1.33470547 1.33469129]
    168 dia. Valor Previsto -> [[1.334678]]
    169 dia. Valores de Entrada -> [1.33447158 1.33447099 1.33447826 1.33449256 1.33451259 1.33453786
     1.33456612 1.33459711 1.33462906 1.33466089 1.33469117 1.33471978
     1.33474565 1.33476758 1.33478582 1.33479989 1.33480859 1.33481383
     1.33481431 1.33481133 1.33480453 1.33479476 1.33478284 1.33476853
     1.33475304 1.33473718 1.33472109 1.33470547 1.33469129 1.33467805]
    169 dia. Valor Previsto -> [[1.3346666]]
    170 dia. Valores de Entrada -> [1.33447099 1.33447826 1.33449256 1.33451259 1.33453786 1.33456612
     1.33459711 1.33462906 1.33466089 1.33469117 1.33471978 1.33474565
     1.33476758 1.33478582 1.33479989 1.33480859 1.33481383 1.33481431
     1.33481133 1.33480453 1.33479476 1.33478284 1.33476853 1.33475304
     1.33473718 1.33472109 1.33470547 1.33469129 1.33467805 1.33466661]
    170 dia. Valor Previsto -> [[1.3346573]]
    171 dia. Valores de Entrada -> [1.33447826 1.33449256 1.33451259 1.33453786 1.33456612 1.33459711
     1.33462906 1.33466089 1.33469117 1.33471978 1.33474565 1.33476758
     1.33478582 1.33479989 1.33480859 1.33481383 1.33481431 1.33481133
     1.33480453 1.33479476 1.33478284 1.33476853 1.33475304 1.33473718
     1.33472109 1.33470547 1.33469129 1.33467805 1.33466661 1.33465731]
    171 dia. Valor Previsto -> [[1.3346502]]
    172 dia. Valores de Entrada -> [1.33449256 1.33451259 1.33453786 1.33456612 1.33459711 1.33462906
     1.33466089 1.33469117 1.33471978 1.33474565 1.33476758 1.33478582
     1.33479989 1.33480859 1.33481383 1.33481431 1.33481133 1.33480453
     1.33479476 1.33478284 1.33476853 1.33475304 1.33473718 1.33472109
     1.33470547 1.33469129 1.33467805 1.33466661 1.33465731 1.33465016]
    172 dia. Valor Previsto -> [[1.3346452]]
    173 dia. Valores de Entrada -> [1.33451259 1.33453786 1.33456612 1.33459711 1.33462906 1.33466089
     1.33469117 1.33471978 1.33474565 1.33476758 1.33478582 1.33479989
     1.33480859 1.33481383 1.33481431 1.33481133 1.33480453 1.33479476
     1.33478284 1.33476853 1.33475304 1.33473718 1.33472109 1.33470547
     1.33469129 1.33467805 1.33466661 1.33465731 1.33465016 1.33464515]
    173 dia. Valor Previsto -> [[1.3346424]]
    174 dia. Valores de Entrada -> [1.33453786 1.33456612 1.33459711 1.33462906 1.33466089 1.33469117
     1.33471978 1.33474565 1.33476758 1.33478582 1.33479989 1.33480859
     1.33481383 1.33481431 1.33481133 1.33480453 1.33479476 1.33478284
     1.33476853 1.33475304 1.33473718 1.33472109 1.33470547 1.33469129
     1.33467805 1.33466661 1.33465731 1.33465016 1.33464515 1.33464241]
    174 dia. Valor Previsto -> [[1.3346418]]
    175 dia. Valores de Entrada -> [1.33456612 1.33459711 1.33462906 1.33466089 1.33469117 1.33471978
     1.33474565 1.33476758 1.33478582 1.33479989 1.33480859 1.33481383
     1.33481431 1.33481133 1.33480453 1.33479476 1.33478284 1.33476853
     1.33475304 1.33473718 1.33472109 1.33470547 1.33469129 1.33467805
     1.33466661 1.33465731 1.33465016 1.33464515 1.33464241 1.33464181]
    175 dia. Valor Previsto -> [[1.334643]]
    176 dia. Valores de Entrada -> [1.33459711 1.33462906 1.33466089 1.33469117 1.33471978 1.33474565
     1.33476758 1.33478582 1.33479989 1.33480859 1.33481383 1.33481431
     1.33481133 1.33480453 1.33479476 1.33478284 1.33476853 1.33475304
     1.33473718 1.33472109 1.33470547 1.33469129 1.33467805 1.33466661
     1.33465731 1.33465016 1.33464515 1.33464241 1.33464181 1.33464301]
    176 dia. Valor Previsto -> [[1.3346463]]
    177 dia. Valores de Entrada -> [1.33462906 1.33466089 1.33469117 1.33471978 1.33474565 1.33476758
     1.33478582 1.33479989 1.33480859 1.33481383 1.33481431 1.33481133
     1.33480453 1.33479476 1.33478284 1.33476853 1.33475304 1.33473718
     1.33472109 1.33470547 1.33469129 1.33467805 1.33466661 1.33465731
     1.33465016 1.33464515 1.33464241 1.33464181 1.33464301 1.33464634]
    177 dia. Valor Previsto -> [[1.3346514]]
    178 dia. Valores de Entrada -> [1.33466089 1.33469117 1.33471978 1.33474565 1.33476758 1.33478582
     1.33479989 1.33480859 1.33481383 1.33481431 1.33481133 1.33480453
     1.33479476 1.33478284 1.33476853 1.33475304 1.33473718 1.33472109
     1.33470547 1.33469129 1.33467805 1.33466661 1.33465731 1.33465016
     1.33464515 1.33464241 1.33464181 1.33464301 1.33464634 1.33465135]
    178 dia. Valor Previsto -> [[1.3346571]]
    179 dia. Valores de Entrada -> [1.33469117 1.33471978 1.33474565 1.33476758 1.33478582 1.33479989
     1.33480859 1.33481383 1.33481431 1.33481133 1.33480453 1.33479476
     1.33478284 1.33476853 1.33475304 1.33473718 1.33472109 1.33470547
     1.33469129 1.33467805 1.33466661 1.33465731 1.33465016 1.33464515
     1.33464241 1.33464181 1.33464301 1.33464634 1.33465135 1.33465707]
    179 dia. Valor Previsto -> [[1.334664]]
    180 dia. Valores de Entrada -> [1.33471978 1.33474565 1.33476758 1.33478582 1.33479989 1.33480859
     1.33481383 1.33481431 1.33481133 1.33480453 1.33479476 1.33478284
     1.33476853 1.33475304 1.33473718 1.33472109 1.33470547 1.33469129
     1.33467805 1.33466661 1.33465731 1.33465016 1.33464515 1.33464241
     1.33464181 1.33464301 1.33464634 1.33465135 1.33465707 1.33466399]
    180 dia. Valor Previsto -> [[1.3346717]]
    181 dia. Valores de Entrada -> [1.33474565 1.33476758 1.33478582 1.33479989 1.33480859 1.33481383
     1.33481431 1.33481133 1.33480453 1.33479476 1.33478284 1.33476853
     1.33475304 1.33473718 1.33472109 1.33470547 1.33469129 1.33467805
     1.33466661 1.33465731 1.33465016 1.33464515 1.33464241 1.33464181
     1.33464301 1.33464634 1.33465135 1.33465707 1.33466399 1.33467174]
    181 dia. Valor Previsto -> [[1.3346795]]
    182 dia. Valores de Entrada -> [1.33476758 1.33478582 1.33479989 1.33480859 1.33481383 1.33481431
     1.33481133 1.33480453 1.33479476 1.33478284 1.33476853 1.33475304
     1.33473718 1.33472109 1.33470547 1.33469129 1.33467805 1.33466661
     1.33465731 1.33465016 1.33464515 1.33464241 1.33464181 1.33464301
     1.33464634 1.33465135 1.33465707 1.33466399 1.33467174 1.33467948]
    182 dia. Valor Previsto -> [[1.3346877]]
    183 dia. Valores de Entrada -> [1.33478582 1.33479989 1.33480859 1.33481383 1.33481431 1.33481133
     1.33480453 1.33479476 1.33478284 1.33476853 1.33475304 1.33473718
     1.33472109 1.33470547 1.33469129 1.33467805 1.33466661 1.33465731
     1.33465016 1.33464515 1.33464241 1.33464181 1.33464301 1.33464634
     1.33465135 1.33465707 1.33466399 1.33467174 1.33467948 1.33468771]
    183 dia. Valor Previsto -> [[1.3346956]]
    184 dia. Valores de Entrada -> [1.33479989 1.33480859 1.33481383 1.33481431 1.33481133 1.33480453
     1.33479476 1.33478284 1.33476853 1.33475304 1.33473718 1.33472109
     1.33470547 1.33469129 1.33467805 1.33466661 1.33465731 1.33465016
     1.33464515 1.33464241 1.33464181 1.33464301 1.33464634 1.33465135
     1.33465707 1.33466399 1.33467174 1.33467948 1.33468771 1.33469558]
    184 dia. Valor Previsto -> [[1.3347031]]
    185 dia. Valores de Entrada -> [1.33480859 1.33481383 1.33481431 1.33481133 1.33480453 1.33479476
     1.33478284 1.33476853 1.33475304 1.33473718 1.33472109 1.33470547
     1.33469129 1.33467805 1.33466661 1.33465731 1.33465016 1.33464515
     1.33464241 1.33464181 1.33464301 1.33464634 1.33465135 1.33465707
     1.33466399 1.33467174 1.33467948 1.33468771 1.33469558 1.33470309]
    185 dia. Valor Previsto -> [[1.3347096]]
    186 dia. Valores de Entrada -> [1.33481383 1.33481431 1.33481133 1.33480453 1.33479476 1.33478284
     1.33476853 1.33475304 1.33473718 1.33472109 1.33470547 1.33469129
     1.33467805 1.33466661 1.33465731 1.33465016 1.33464515 1.33464241
     1.33464181 1.33464301 1.33464634 1.33465135 1.33465707 1.33466399
     1.33467174 1.33467948 1.33468771 1.33469558 1.33470309 1.33470964]
    186 dia. Valor Previsto -> [[1.3347157]]
    187 dia. Valores de Entrada -> [1.33481431 1.33481133 1.33480453 1.33479476 1.33478284 1.33476853
     1.33475304 1.33473718 1.33472109 1.33470547 1.33469129 1.33467805
     1.33466661 1.33465731 1.33465016 1.33464515 1.33464241 1.33464181
     1.33464301 1.33464634 1.33465135 1.33465707 1.33466399 1.33467174
     1.33467948 1.33468771 1.33469558 1.33470309 1.33470964 1.33471572]
    187 dia. Valor Previsto -> [[1.3347207]]
    188 dia. Valores de Entrada -> [1.33481133 1.33480453 1.33479476 1.33478284 1.33476853 1.33475304
     1.33473718 1.33472109 1.33470547 1.33469129 1.33467805 1.33466661
     1.33465731 1.33465016 1.33464515 1.33464241 1.33464181 1.33464301
     1.33464634 1.33465135 1.33465707 1.33466399 1.33467174 1.33467948
     1.33468771 1.33469558 1.33470309 1.33470964 1.33471572 1.33472073]
    188 dia. Valor Previsto -> [[1.3347243]]
    189 dia. Valores de Entrada -> [1.33480453 1.33479476 1.33478284 1.33476853 1.33475304 1.33473718
     1.33472109 1.33470547 1.33469129 1.33467805 1.33466661 1.33465731
     1.33465016 1.33464515 1.33464241 1.33464181 1.33464301 1.33464634
     1.33465135 1.33465707 1.33466399 1.33467174 1.33467948 1.33468771
     1.33469558 1.33470309 1.33470964 1.33471572 1.33472073 1.33472431]
    189 dia. Valor Previsto -> [[1.334727]]
    190 dia. Valores de Entrada -> [1.33479476 1.33478284 1.33476853 1.33475304 1.33473718 1.33472109
     1.33470547 1.33469129 1.33467805 1.33466661 1.33465731 1.33465016
     1.33464515 1.33464241 1.33464181 1.33464301 1.33464634 1.33465135
     1.33465707 1.33466399 1.33467174 1.33467948 1.33468771 1.33469558
     1.33470309 1.33470964 1.33471572 1.33472073 1.33472431 1.33472705]
    190 dia. Valor Previsto -> [[1.3347288]]
    191 dia. Valores de Entrada -> [1.33478284 1.33476853 1.33475304 1.33473718 1.33472109 1.33470547
     1.33469129 1.33467805 1.33466661 1.33465731 1.33465016 1.33464515
     1.33464241 1.33464181 1.33464301 1.33464634 1.33465135 1.33465707
     1.33466399 1.33467174 1.33467948 1.33468771 1.33469558 1.33470309
     1.33470964 1.33471572 1.33472073 1.33472431 1.33472705 1.33472884]
    191 dia. Valor Previsto -> [[1.3347293]]
    192 dia. Valores de Entrada -> [1.33476853 1.33475304 1.33473718 1.33472109 1.33470547 1.33469129
     1.33467805 1.33466661 1.33465731 1.33465016 1.33464515 1.33464241
     1.33464181 1.33464301 1.33464634 1.33465135 1.33465707 1.33466399
     1.33467174 1.33467948 1.33468771 1.33469558 1.33470309 1.33470964
     1.33471572 1.33472073 1.33472431 1.33472705 1.33472884 1.33472931]
    192 dia. Valor Previsto -> [[1.3347288]]
    193 dia. Valores de Entrada -> [1.33475304 1.33473718 1.33472109 1.33470547 1.33469129 1.33467805
     1.33466661 1.33465731 1.33465016 1.33464515 1.33464241 1.33464181
     1.33464301 1.33464634 1.33465135 1.33465707 1.33466399 1.33467174
     1.33467948 1.33468771 1.33469558 1.33470309 1.33470964 1.33471572
     1.33472073 1.33472431 1.33472705 1.33472884 1.33472931 1.33472884]
    193 dia. Valor Previsto -> [[1.334727]]
    194 dia. Valores de Entrada -> [1.33473718 1.33472109 1.33470547 1.33469129 1.33467805 1.33466661
     1.33465731 1.33465016 1.33464515 1.33464241 1.33464181 1.33464301
     1.33464634 1.33465135 1.33465707 1.33466399 1.33467174 1.33467948
     1.33468771 1.33469558 1.33470309 1.33470964 1.33471572 1.33472073
     1.33472431 1.33472705 1.33472884 1.33472931 1.33472884 1.33472705]
    194 dia. Valor Previsto -> [[1.3347248]]
    195 dia. Valores de Entrada -> [1.33472109 1.33470547 1.33469129 1.33467805 1.33466661 1.33465731
     1.33465016 1.33464515 1.33464241 1.33464181 1.33464301 1.33464634
     1.33465135 1.33465707 1.33466399 1.33467174 1.33467948 1.33468771
     1.33469558 1.33470309 1.33470964 1.33471572 1.33472073 1.33472431
     1.33472705 1.33472884 1.33472931 1.33472884 1.33472705 1.33472478]
    195 dia. Valor Previsto -> [[1.3347213]]
    196 dia. Valores de Entrada -> [1.33470547 1.33469129 1.33467805 1.33466661 1.33465731 1.33465016
     1.33464515 1.33464241 1.33464181 1.33464301 1.33464634 1.33465135
     1.33465707 1.33466399 1.33467174 1.33467948 1.33468771 1.33469558
     1.33470309 1.33470964 1.33471572 1.33472073 1.33472431 1.33472705
     1.33472884 1.33472931 1.33472884 1.33472705 1.33472478 1.33472133]
    196 dia. Valor Previsto -> [[1.3347181]]
    197 dia. Valores de Entrada -> [1.33469129 1.33467805 1.33466661 1.33465731 1.33465016 1.33464515
     1.33464241 1.33464181 1.33464301 1.33464634 1.33465135 1.33465707
     1.33466399 1.33467174 1.33467948 1.33468771 1.33469558 1.33470309
     1.33470964 1.33471572 1.33472073 1.33472431 1.33472705 1.33472884
     1.33472931 1.33472884 1.33472705 1.33472478 1.33472133 1.33471811]
    197 dia. Valor Previsto -> [[1.334714]]
    198 dia. Valores de Entrada -> [1.33467805 1.33466661 1.33465731 1.33465016 1.33464515 1.33464241
     1.33464181 1.33464301 1.33464634 1.33465135 1.33465707 1.33466399
     1.33467174 1.33467948 1.33468771 1.33469558 1.33470309 1.33470964
     1.33471572 1.33472073 1.33472431 1.33472705 1.33472884 1.33472931
     1.33472884 1.33472705 1.33472478 1.33472133 1.33471811 1.33471406]
    198 dia. Valor Previsto -> [[1.33471]]
    199 dia. Valores de Entrada -> [1.33466661 1.33465731 1.33465016 1.33464515 1.33464241 1.33464181
     1.33464301 1.33464634 1.33465135 1.33465707 1.33466399 1.33467174
     1.33467948 1.33468771 1.33469558 1.33470309 1.33470964 1.33471572
     1.33472073 1.33472431 1.33472705 1.33472884 1.33472931 1.33472884
     1.33472705 1.33472478 1.33472133 1.33471811 1.33471406 1.33471   ]
    199 dia. Valor Previsto -> [[1.3347057]]
    200 dia. Valores de Entrada -> [1.33465731 1.33465016 1.33464515 1.33464241 1.33464181 1.33464301
     1.33464634 1.33465135 1.33465707 1.33466399 1.33467174 1.33467948
     1.33468771 1.33469558 1.33470309 1.33470964 1.33471572 1.33472073
     1.33472431 1.33472705 1.33472884 1.33472931 1.33472884 1.33472705
     1.33472478 1.33472133 1.33471811 1.33471406 1.33471    1.33470571]
    200 dia. Valor Previsto -> [[1.3347019]]
    201 dia. Valores de Entrada -> [1.33465016 1.33464515 1.33464241 1.33464181 1.33464301 1.33464634
     1.33465135 1.33465707 1.33466399 1.33467174 1.33467948 1.33468771
     1.33469558 1.33470309 1.33470964 1.33471572 1.33472073 1.33472431
     1.33472705 1.33472884 1.33472931 1.33472884 1.33472705 1.33472478
     1.33472133 1.33471811 1.33471406 1.33471    1.33470571 1.3347019 ]
    201 dia. Valor Previsto -> [[1.3346981]]
    202 dia. Valores de Entrada -> [1.33464515 1.33464241 1.33464181 1.33464301 1.33464634 1.33465135
     1.33465707 1.33466399 1.33467174 1.33467948 1.33468771 1.33469558
     1.33470309 1.33470964 1.33471572 1.33472073 1.33472431 1.33472705
     1.33472884 1.33472931 1.33472884 1.33472705 1.33472478 1.33472133
     1.33471811 1.33471406 1.33471    1.33470571 1.3347019  1.33469808]
    202 dia. Valor Previsto -> [[1.3346946]]
    203 dia. Valores de Entrada -> [1.33464241 1.33464181 1.33464301 1.33464634 1.33465135 1.33465707
     1.33466399 1.33467174 1.33467948 1.33468771 1.33469558 1.33470309
     1.33470964 1.33471572 1.33472073 1.33472431 1.33472705 1.33472884
     1.33472931 1.33472884 1.33472705 1.33472478 1.33472133 1.33471811
     1.33471406 1.33471    1.33470571 1.3347019  1.33469808 1.33469462]
    203 dia. Valor Previsto -> [[1.334692]]
    204 dia. Valores de Entrada -> [1.33464181 1.33464301 1.33464634 1.33465135 1.33465707 1.33466399
     1.33467174 1.33467948 1.33468771 1.33469558 1.33470309 1.33470964
     1.33471572 1.33472073 1.33472431 1.33472705 1.33472884 1.33472931
     1.33472884 1.33472705 1.33472478 1.33472133 1.33471811 1.33471406
     1.33471    1.33470571 1.3347019  1.33469808 1.33469462 1.334692  ]
    204 dia. Valor Previsto -> [[1.334689]]
    205 dia. Valores de Entrada -> [1.33464301 1.33464634 1.33465135 1.33465707 1.33466399 1.33467174
     1.33467948 1.33468771 1.33469558 1.33470309 1.33470964 1.33471572
     1.33472073 1.33472431 1.33472705 1.33472884 1.33472931 1.33472884
     1.33472705 1.33472478 1.33472133 1.33471811 1.33471406 1.33471
     1.33470571 1.3347019  1.33469808 1.33469462 1.334692   1.33468902]
    205 dia. Valor Previsto -> [[1.3346872]]
    206 dia. Valores de Entrada -> [1.33464634 1.33465135 1.33465707 1.33466399 1.33467174 1.33467948
     1.33468771 1.33469558 1.33470309 1.33470964 1.33471572 1.33472073
     1.33472431 1.33472705 1.33472884 1.33472931 1.33472884 1.33472705
     1.33472478 1.33472133 1.33471811 1.33471406 1.33471    1.33470571
     1.3347019  1.33469808 1.33469462 1.334692   1.33468902 1.33468723]
    206 dia. Valor Previsto -> [[1.3346858]]
    207 dia. Valores de Entrada -> [1.33465135 1.33465707 1.33466399 1.33467174 1.33467948 1.33468771
     1.33469558 1.33470309 1.33470964 1.33471572 1.33472073 1.33472431
     1.33472705 1.33472884 1.33472931 1.33472884 1.33472705 1.33472478
     1.33472133 1.33471811 1.33471406 1.33471    1.33470571 1.3347019
     1.33469808 1.33469462 1.334692   1.33468902 1.33468723 1.3346858 ]
    207 dia. Valor Previsto -> [[1.3346852]]
    208 dia. Valores de Entrada -> [1.33465707 1.33466399 1.33467174 1.33467948 1.33468771 1.33469558
     1.33470309 1.33470964 1.33471572 1.33472073 1.33472431 1.33472705
     1.33472884 1.33472931 1.33472884 1.33472705 1.33472478 1.33472133
     1.33471811 1.33471406 1.33471    1.33470571 1.3347019  1.33469808
     1.33469462 1.334692   1.33468902 1.33468723 1.3346858  1.33468521]
    208 dia. Valor Previsto -> [[1.3346847]]
    209 dia. Valores de Entrada -> [1.33466399 1.33467174 1.33467948 1.33468771 1.33469558 1.33470309
     1.33470964 1.33471572 1.33472073 1.33472431 1.33472705 1.33472884
     1.33472931 1.33472884 1.33472705 1.33472478 1.33472133 1.33471811
     1.33471406 1.33471    1.33470571 1.3347019  1.33469808 1.33469462
     1.334692   1.33468902 1.33468723 1.3346858  1.33468521 1.33468473]
    209 dia. Valor Previsto -> [[1.3346852]]
    210 dia. Valores de Entrada -> [1.33467174 1.33467948 1.33468771 1.33469558 1.33470309 1.33470964
     1.33471572 1.33472073 1.33472431 1.33472705 1.33472884 1.33472931
     1.33472884 1.33472705 1.33472478 1.33472133 1.33471811 1.33471406
     1.33471    1.33470571 1.3347019  1.33469808 1.33469462 1.334692
     1.33468902 1.33468723 1.3346858  1.33468521 1.33468473 1.33468521]
    210 dia. Valor Previsto -> [[1.3346856]]
    211 dia. Valores de Entrada -> [1.33467948 1.33468771 1.33469558 1.33470309 1.33470964 1.33471572
     1.33472073 1.33472431 1.33472705 1.33472884 1.33472931 1.33472884
     1.33472705 1.33472478 1.33472133 1.33471811 1.33471406 1.33471
     1.33470571 1.3347019  1.33469808 1.33469462 1.334692   1.33468902
     1.33468723 1.3346858  1.33468521 1.33468473 1.33468521 1.33468556]
    211 dia. Valor Previsto -> [[1.3346869]]
    212 dia. Valores de Entrada -> [1.33468771 1.33469558 1.33470309 1.33470964 1.33471572 1.33472073
     1.33472431 1.33472705 1.33472884 1.33472931 1.33472884 1.33472705
     1.33472478 1.33472133 1.33471811 1.33471406 1.33471    1.33470571
     1.3347019  1.33469808 1.33469462 1.334692   1.33468902 1.33468723
     1.3346858  1.33468521 1.33468473 1.33468521 1.33468556 1.33468688]
    212 dia. Valor Previsto -> [[1.3346885]]
    213 dia. Valores de Entrada -> [1.33469558 1.33470309 1.33470964 1.33471572 1.33472073 1.33472431
     1.33472705 1.33472884 1.33472931 1.33472884 1.33472705 1.33472478
     1.33472133 1.33471811 1.33471406 1.33471    1.33470571 1.3347019
     1.33469808 1.33469462 1.334692   1.33468902 1.33468723 1.3346858
     1.33468521 1.33468473 1.33468521 1.33468556 1.33468688 1.33468854]
    213 dia. Valor Previsto -> [[1.3346907]]
    214 dia. Valores de Entrada -> [1.33470309 1.33470964 1.33471572 1.33472073 1.33472431 1.33472705
     1.33472884 1.33472931 1.33472884 1.33472705 1.33472478 1.33472133
     1.33471811 1.33471406 1.33471    1.33470571 1.3347019  1.33469808
     1.33469462 1.334692   1.33468902 1.33468723 1.3346858  1.33468521
     1.33468473 1.33468521 1.33468556 1.33468688 1.33468854 1.33469069]
    214 dia. Valor Previsto -> [[1.3346926]]
    215 dia. Valores de Entrada -> [1.33470964 1.33471572 1.33472073 1.33472431 1.33472705 1.33472884
     1.33472931 1.33472884 1.33472705 1.33472478 1.33472133 1.33471811
     1.33471406 1.33471    1.33470571 1.3347019  1.33469808 1.33469462
     1.334692   1.33468902 1.33468723 1.3346858  1.33468521 1.33468473
     1.33468521 1.33468556 1.33468688 1.33468854 1.33469069 1.3346926 ]
    215 dia. Valor Previsto -> [[1.3346946]]
    216 dia. Valores de Entrada -> [1.33471572 1.33472073 1.33472431 1.33472705 1.33472884 1.33472931
     1.33472884 1.33472705 1.33472478 1.33472133 1.33471811 1.33471406
     1.33471    1.33470571 1.3347019  1.33469808 1.33469462 1.334692
     1.33468902 1.33468723 1.3346858  1.33468521 1.33468473 1.33468521
     1.33468556 1.33468688 1.33468854 1.33469069 1.3346926  1.33469462]
    216 dia. Valor Previsto -> [[1.3346964]]
    217 dia. Valores de Entrada -> [1.33472073 1.33472431 1.33472705 1.33472884 1.33472931 1.33472884
     1.33472705 1.33472478 1.33472133 1.33471811 1.33471406 1.33471
     1.33470571 1.3347019  1.33469808 1.33469462 1.334692   1.33468902
     1.33468723 1.3346858  1.33468521 1.33468473 1.33468521 1.33468556
     1.33468688 1.33468854 1.33469069 1.3346926  1.33469462 1.33469641]
    217 dia. Valor Previsto -> [[1.3346986]]
    218 dia. Valores de Entrada -> [1.33472431 1.33472705 1.33472884 1.33472931 1.33472884 1.33472705
     1.33472478 1.33472133 1.33471811 1.33471406 1.33471    1.33470571
     1.3347019  1.33469808 1.33469462 1.334692   1.33468902 1.33468723
     1.3346858  1.33468521 1.33468473 1.33468521 1.33468556 1.33468688
     1.33468854 1.33469069 1.3346926  1.33469462 1.33469641 1.33469856]
    218 dia. Valor Previsto -> [[1.3347002]]
    219 dia. Valores de Entrada -> [1.33472705 1.33472884 1.33472931 1.33472884 1.33472705 1.33472478
     1.33472133 1.33471811 1.33471406 1.33471    1.33470571 1.3347019
     1.33469808 1.33469462 1.334692   1.33468902 1.33468723 1.3346858
     1.33468521 1.33468473 1.33468521 1.33468556 1.33468688 1.33468854
     1.33469069 1.3346926  1.33469462 1.33469641 1.33469856 1.33470023]
    219 dia. Valor Previsto -> [[1.334702]]
    220 dia. Valores de Entrada -> [1.33472884 1.33472931 1.33472884 1.33472705 1.33472478 1.33472133
     1.33471811 1.33471406 1.33471    1.33470571 1.3347019  1.33469808
     1.33469462 1.334692   1.33468902 1.33468723 1.3346858  1.33468521
     1.33468473 1.33468521 1.33468556 1.33468688 1.33468854 1.33469069
     1.3346926  1.33469462 1.33469641 1.33469856 1.33470023 1.33470201]
    220 dia. Valor Previsto -> [[1.3347036]]
    221 dia. Valores de Entrada -> [1.33472931 1.33472884 1.33472705 1.33472478 1.33472133 1.33471811
     1.33471406 1.33471    1.33470571 1.3347019  1.33469808 1.33469462
     1.334692   1.33468902 1.33468723 1.3346858  1.33468521 1.33468473
     1.33468521 1.33468556 1.33468688 1.33468854 1.33469069 1.3346926
     1.33469462 1.33469641 1.33469856 1.33470023 1.33470201 1.33470356]
    221 dia. Valor Previsto -> [[1.3347049]]
    222 dia. Valores de Entrada -> [1.33472884 1.33472705 1.33472478 1.33472133 1.33471811 1.33471406
     1.33471    1.33470571 1.3347019  1.33469808 1.33469462 1.334692
     1.33468902 1.33468723 1.3346858  1.33468521 1.33468473 1.33468521
     1.33468556 1.33468688 1.33468854 1.33469069 1.3346926  1.33469462
     1.33469641 1.33469856 1.33470023 1.33470201 1.33470356 1.33470488]
    222 dia. Valor Previsto -> [[1.3347058]]
    223 dia. Valores de Entrada -> [1.33472705 1.33472478 1.33472133 1.33471811 1.33471406 1.33471
     1.33470571 1.3347019  1.33469808 1.33469462 1.334692   1.33468902
     1.33468723 1.3346858  1.33468521 1.33468473 1.33468521 1.33468556
     1.33468688 1.33468854 1.33469069 1.3346926  1.33469462 1.33469641
     1.33469856 1.33470023 1.33470201 1.33470356 1.33470488 1.33470583]
    223 dia. Valor Previsto -> [[1.3347065]]
    224 dia. Valores de Entrada -> [1.33472478 1.33472133 1.33471811 1.33471406 1.33471    1.33470571
     1.3347019  1.33469808 1.33469462 1.334692   1.33468902 1.33468723
     1.3346858  1.33468521 1.33468473 1.33468521 1.33468556 1.33468688
     1.33468854 1.33469069 1.3346926  1.33469462 1.33469641 1.33469856
     1.33470023 1.33470201 1.33470356 1.33470488 1.33470583 1.33470654]
    224 dia. Valor Previsto -> [[1.3347068]]
    225 dia. Valores de Entrada -> [1.33472133 1.33471811 1.33471406 1.33471    1.33470571 1.3347019
     1.33469808 1.33469462 1.334692   1.33468902 1.33468723 1.3346858
     1.33468521 1.33468473 1.33468521 1.33468556 1.33468688 1.33468854
     1.33469069 1.3346926  1.33469462 1.33469641 1.33469856 1.33470023
     1.33470201 1.33470356 1.33470488 1.33470583 1.33470654 1.33470678]
    225 dia. Valor Previsto -> [[1.3347069]]
    226 dia. Valores de Entrada -> [1.33471811 1.33471406 1.33471    1.33470571 1.3347019  1.33469808
     1.33469462 1.334692   1.33468902 1.33468723 1.3346858  1.33468521
     1.33468473 1.33468521 1.33468556 1.33468688 1.33468854 1.33469069
     1.3346926  1.33469462 1.33469641 1.33469856 1.33470023 1.33470201
     1.33470356 1.33470488 1.33470583 1.33470654 1.33470678 1.3347069 ]
    226 dia. Valor Previsto -> [[1.3347065]]
    227 dia. Valores de Entrada -> [1.33471406 1.33471    1.33470571 1.3347019  1.33469808 1.33469462
     1.334692   1.33468902 1.33468723 1.3346858  1.33468521 1.33468473
     1.33468521 1.33468556 1.33468688 1.33468854 1.33469069 1.3346926
     1.33469462 1.33469641 1.33469856 1.33470023 1.33470201 1.33470356
     1.33470488 1.33470583 1.33470654 1.33470678 1.3347069  1.33470654]
    227 dia. Valor Previsto -> [[1.3347063]]
    228 dia. Valores de Entrada -> [1.33471    1.33470571 1.3347019  1.33469808 1.33469462 1.334692
     1.33468902 1.33468723 1.3346858  1.33468521 1.33468473 1.33468521
     1.33468556 1.33468688 1.33468854 1.33469069 1.3346926  1.33469462
     1.33469641 1.33469856 1.33470023 1.33470201 1.33470356 1.33470488
     1.33470583 1.33470654 1.33470678 1.3347069  1.33470654 1.33470631]
    228 dia. Valor Previsto -> [[1.334706]]
    229 dia. Valores de Entrada -> [1.33470571 1.3347019  1.33469808 1.33469462 1.334692   1.33468902
     1.33468723 1.3346858  1.33468521 1.33468473 1.33468521 1.33468556
     1.33468688 1.33468854 1.33469069 1.3346926  1.33469462 1.33469641
     1.33469856 1.33470023 1.33470201 1.33470356 1.33470488 1.33470583
     1.33470654 1.33470678 1.3347069  1.33470654 1.33470631 1.33470595]
    229 dia. Valor Previsto -> [[1.3347052]]
    230 dia. Valores de Entrada -> [1.3347019  1.33469808 1.33469462 1.334692   1.33468902 1.33468723
     1.3346858  1.33468521 1.33468473 1.33468521 1.33468556 1.33468688
     1.33468854 1.33469069 1.3346926  1.33469462 1.33469641 1.33469856
     1.33470023 1.33470201 1.33470356 1.33470488 1.33470583 1.33470654
     1.33470678 1.3347069  1.33470654 1.33470631 1.33470595 1.33470523]
    230 dia. Valor Previsto -> [[1.3347042]]
    231 dia. Valores de Entrada -> [1.33469808 1.33469462 1.334692   1.33468902 1.33468723 1.3346858
     1.33468521 1.33468473 1.33468521 1.33468556 1.33468688 1.33468854
     1.33469069 1.3346926  1.33469462 1.33469641 1.33469856 1.33470023
     1.33470201 1.33470356 1.33470488 1.33470583 1.33470654 1.33470678
     1.3347069  1.33470654 1.33470631 1.33470595 1.33470523 1.33470416]
    231 dia. Valor Previsto -> [[1.3347032]]
    232 dia. Valores de Entrada -> [1.33469462 1.334692   1.33468902 1.33468723 1.3346858  1.33468521
     1.33468473 1.33468521 1.33468556 1.33468688 1.33468854 1.33469069
     1.3346926  1.33469462 1.33469641 1.33469856 1.33470023 1.33470201
     1.33470356 1.33470488 1.33470583 1.33470654 1.33470678 1.3347069
     1.33470654 1.33470631 1.33470595 1.33470523 1.33470416 1.33470321]
    232 dia. Valor Previsto -> [[1.334702]]
    233 dia. Valores de Entrada -> [1.334692   1.33468902 1.33468723 1.3346858  1.33468521 1.33468473
     1.33468521 1.33468556 1.33468688 1.33468854 1.33469069 1.3346926
     1.33469462 1.33469641 1.33469856 1.33470023 1.33470201 1.33470356
     1.33470488 1.33470583 1.33470654 1.33470678 1.3347069  1.33470654
     1.33470631 1.33470595 1.33470523 1.33470416 1.33470321 1.33470201]
    233 dia. Valor Previsto -> [[1.3347012]]
    234 dia. Valores de Entrada -> [1.33468902 1.33468723 1.3346858  1.33468521 1.33468473 1.33468521
     1.33468556 1.33468688 1.33468854 1.33469069 1.3346926  1.33469462
     1.33469641 1.33469856 1.33470023 1.33470201 1.33470356 1.33470488
     1.33470583 1.33470654 1.33470678 1.3347069  1.33470654 1.33470631
     1.33470595 1.33470523 1.33470416 1.33470321 1.33470201 1.33470118]
    234 dia. Valor Previsto -> [[1.3347003]]
    235 dia. Valores de Entrada -> [1.33468723 1.3346858  1.33468521 1.33468473 1.33468521 1.33468556
     1.33468688 1.33468854 1.33469069 1.3346926  1.33469462 1.33469641
     1.33469856 1.33470023 1.33470201 1.33470356 1.33470488 1.33470583
     1.33470654 1.33470678 1.3347069  1.33470654 1.33470631 1.33470595
     1.33470523 1.33470416 1.33470321 1.33470201 1.33470118 1.33470035]
    235 dia. Valor Previsto -> [[1.3346993]]
    236 dia. Valores de Entrada -> [1.3346858  1.33468521 1.33468473 1.33468521 1.33468556 1.33468688
     1.33468854 1.33469069 1.3346926  1.33469462 1.33469641 1.33469856
     1.33470023 1.33470201 1.33470356 1.33470488 1.33470583 1.33470654
     1.33470678 1.3347069  1.33470654 1.33470631 1.33470595 1.33470523
     1.33470416 1.33470321 1.33470201 1.33470118 1.33470035 1.33469927]
    236 dia. Valor Previsto -> [[1.3346986]]
    237 dia. Valores de Entrada -> [1.33468521 1.33468473 1.33468521 1.33468556 1.33468688 1.33468854
     1.33469069 1.3346926  1.33469462 1.33469641 1.33469856 1.33470023
     1.33470201 1.33470356 1.33470488 1.33470583 1.33470654 1.33470678
     1.3347069  1.33470654 1.33470631 1.33470595 1.33470523 1.33470416
     1.33470321 1.33470201 1.33470118 1.33470035 1.33469927 1.33469856]
    237 dia. Valor Previsto -> [[1.3346978]]
    238 dia. Valores de Entrada -> [1.33468473 1.33468521 1.33468556 1.33468688 1.33468854 1.33469069
     1.3346926  1.33469462 1.33469641 1.33469856 1.33470023 1.33470201
     1.33470356 1.33470488 1.33470583 1.33470654 1.33470678 1.3347069
     1.33470654 1.33470631 1.33470595 1.33470523 1.33470416 1.33470321
     1.33470201 1.33470118 1.33470035 1.33469927 1.33469856 1.33469784]
    238 dia. Valor Previsto -> [[1.3346972]]
    239 dia. Valores de Entrada -> [1.33468521 1.33468556 1.33468688 1.33468854 1.33469069 1.3346926
     1.33469462 1.33469641 1.33469856 1.33470023 1.33470201 1.33470356
     1.33470488 1.33470583 1.33470654 1.33470678 1.3347069  1.33470654
     1.33470631 1.33470595 1.33470523 1.33470416 1.33470321 1.33470201
     1.33470118 1.33470035 1.33469927 1.33469856 1.33469784 1.33469725]
    239 dia. Valor Previsto -> [[1.3346969]]
    240 dia. Valores de Entrada -> [1.33468556 1.33468688 1.33468854 1.33469069 1.3346926  1.33469462
     1.33469641 1.33469856 1.33470023 1.33470201 1.33470356 1.33470488
     1.33470583 1.33470654 1.33470678 1.3347069  1.33470654 1.33470631
     1.33470595 1.33470523 1.33470416 1.33470321 1.33470201 1.33470118
     1.33470035 1.33469927 1.33469856 1.33469784 1.33469725 1.33469689]
    240 dia. Valor Previsto -> [[1.3346967]]
    241 dia. Valores de Entrada -> [1.33468688 1.33468854 1.33469069 1.3346926  1.33469462 1.33469641
     1.33469856 1.33470023 1.33470201 1.33470356 1.33470488 1.33470583
     1.33470654 1.33470678 1.3347069  1.33470654 1.33470631 1.33470595
     1.33470523 1.33470416 1.33470321 1.33470201 1.33470118 1.33470035
     1.33469927 1.33469856 1.33469784 1.33469725 1.33469689 1.33469665]
    241 dia. Valor Previsto -> [[1.334696]]
    242 dia. Valores de Entrada -> [1.33468854 1.33469069 1.3346926  1.33469462 1.33469641 1.33469856
     1.33470023 1.33470201 1.33470356 1.33470488 1.33470583 1.33470654
     1.33470678 1.3347069  1.33470654 1.33470631 1.33470595 1.33470523
     1.33470416 1.33470321 1.33470201 1.33470118 1.33470035 1.33469927
     1.33469856 1.33469784 1.33469725 1.33469689 1.33469665 1.33469605]
    242 dia. Valor Previsto -> [[1.3346958]]
    243 dia. Valores de Entrada -> [1.33469069 1.3346926  1.33469462 1.33469641 1.33469856 1.33470023
     1.33470201 1.33470356 1.33470488 1.33470583 1.33470654 1.33470678
     1.3347069  1.33470654 1.33470631 1.33470595 1.33470523 1.33470416
     1.33470321 1.33470201 1.33470118 1.33470035 1.33469927 1.33469856
     1.33469784 1.33469725 1.33469689 1.33469665 1.33469605 1.33469582]
    243 dia. Valor Previsto -> [[1.3346959]]
    244 dia. Valores de Entrada -> [1.3346926  1.33469462 1.33469641 1.33469856 1.33470023 1.33470201
     1.33470356 1.33470488 1.33470583 1.33470654 1.33470678 1.3347069
     1.33470654 1.33470631 1.33470595 1.33470523 1.33470416 1.33470321
     1.33470201 1.33470118 1.33470035 1.33469927 1.33469856 1.33469784
     1.33469725 1.33469689 1.33469665 1.33469605 1.33469582 1.33469594]
    244 dia. Valor Previsto -> [[1.3346962]]
    245 dia. Valores de Entrada -> [1.33469462 1.33469641 1.33469856 1.33470023 1.33470201 1.33470356
     1.33470488 1.33470583 1.33470654 1.33470678 1.3347069  1.33470654
     1.33470631 1.33470595 1.33470523 1.33470416 1.33470321 1.33470201
     1.33470118 1.33470035 1.33469927 1.33469856 1.33469784 1.33469725
     1.33469689 1.33469665 1.33469605 1.33469582 1.33469594 1.33469617]
    245 dia. Valor Previsto -> [[1.3346967]]
    246 dia. Valores de Entrada -> [1.33469641 1.33469856 1.33470023 1.33470201 1.33470356 1.33470488
     1.33470583 1.33470654 1.33470678 1.3347069  1.33470654 1.33470631
     1.33470595 1.33470523 1.33470416 1.33470321 1.33470201 1.33470118
     1.33470035 1.33469927 1.33469856 1.33469784 1.33469725 1.33469689
     1.33469665 1.33469605 1.33469582 1.33469594 1.33469617 1.33469665]
    246 dia. Valor Previsto -> [[1.3346969]]
    247 dia. Valores de Entrada -> [1.33469856 1.33470023 1.33470201 1.33470356 1.33470488 1.33470583
     1.33470654 1.33470678 1.3347069  1.33470654 1.33470631 1.33470595
     1.33470523 1.33470416 1.33470321 1.33470201 1.33470118 1.33470035
     1.33469927 1.33469856 1.33469784 1.33469725 1.33469689 1.33469665
     1.33469605 1.33469582 1.33469594 1.33469617 1.33469665 1.33469689]
    247 dia. Valor Previsto -> [[1.334697]]
    248 dia. Valores de Entrada -> [1.33470023 1.33470201 1.33470356 1.33470488 1.33470583 1.33470654
     1.33470678 1.3347069  1.33470654 1.33470631 1.33470595 1.33470523
     1.33470416 1.33470321 1.33470201 1.33470118 1.33470035 1.33469927
     1.33469856 1.33469784 1.33469725 1.33469689 1.33469665 1.33469605
     1.33469582 1.33469594 1.33469617 1.33469665 1.33469689 1.33469701]
    248 dia. Valor Previsto -> [[1.3346975]]
    249 dia. Valores de Entrada -> [1.33470201 1.33470356 1.33470488 1.33470583 1.33470654 1.33470678
     1.3347069  1.33470654 1.33470631 1.33470595 1.33470523 1.33470416
     1.33470321 1.33470201 1.33470118 1.33470035 1.33469927 1.33469856
     1.33469784 1.33469725 1.33469689 1.33469665 1.33469605 1.33469582
     1.33469594 1.33469617 1.33469665 1.33469689 1.33469701 1.33469748]
    249 dia. Valor Previsto -> [[1.3346982]]
    250 dia. Valores de Entrada -> [1.33470356 1.33470488 1.33470583 1.33470654 1.33470678 1.3347069
     1.33470654 1.33470631 1.33470595 1.33470523 1.33470416 1.33470321
     1.33470201 1.33470118 1.33470035 1.33469927 1.33469856 1.33469784
     1.33469725 1.33469689 1.33469665 1.33469605 1.33469582 1.33469594
     1.33469617 1.33469665 1.33469689 1.33469701 1.33469748 1.3346982 ]
    250 dia. Valor Previsto -> [[1.3346983]]
    251 dia. Valores de Entrada -> [1.33470488 1.33470583 1.33470654 1.33470678 1.3347069  1.33470654
     1.33470631 1.33470595 1.33470523 1.33470416 1.33470321 1.33470201
     1.33470118 1.33470035 1.33469927 1.33469856 1.33469784 1.33469725
     1.33469689 1.33469665 1.33469605 1.33469582 1.33469594 1.33469617
     1.33469665 1.33469689 1.33469701 1.33469748 1.3346982  1.33469832]
    251 dia. Valor Previsto -> [[1.334699]]
    Previsões -> [[1.2769713401794434], [1.2818434238433838], [1.27913498878479], [1.2785906791687012], [1.2800685167312622], [1.2833219766616821], [1.2881059646606445], [1.2942042350769043], [1.3011442422866821], [1.3086917400360107], [1.3164597749710083], [1.324251651763916], [1.3318990468978882], [1.3390741348266602], [1.3455924987792969], [1.3511778116226196], [1.3557976484298706], [1.3592373132705688], [1.361655592918396], [1.3629121780395508], [1.3630437850952148], [1.3620915412902832], [1.3603452444076538], [1.357883095741272], [1.3548479080200195], [1.3512768745422363], [1.347533941268921], [1.343674898147583], [1.3398391008377075], [1.3361022472381592], [1.3325510025024414], [1.329376220703125], [1.3266031742095947], [1.3243058919906616], [1.322524905204773], [1.3212831020355225], [1.3205828666687012], [1.3204082250595093], [1.3207275867462158], [1.3214936256408691], [1.3226484060287476], [1.324123740196228], [1.3258460760116577], [1.32773756980896], [1.3297202587127686], [1.3317195177078247], [1.3336644172668457], [1.335492730140686], [1.3371508121490479], [1.3385947942733765], [1.3397918939590454], [1.3407212495803833], [1.3413726091384888], [1.3417459726333618], [1.341850996017456], [1.341705560684204], [1.3413344621658325], [1.3407679796218872], [1.3400408029556274], [1.3391907215118408], [1.338254690170288], [1.3372714519500732], [1.336276888847351], [1.335306167602539], [1.3343894481658936], [1.333553433418274], [1.332820177078247], [1.3322062492370605], [1.3317230939865112], [1.3313770294189453], [1.3311691284179688], [1.3310965299606323], [1.3311511278152466], [1.3313215970993042], [1.3315927982330322], [1.3319485187530518], [1.3323700428009033], [1.3328373432159424], [1.3333317041397095], [1.3338342905044556], [1.334327220916748], [1.3347951173782349], [1.3352231979370117], [1.3355998992919922], [1.3359174728393555], [1.3361690044403076], [1.3363511562347412], [1.3364633321762085], [1.3365068435668945], [1.3364864587783813], [1.336406946182251], [1.3362756967544556], [1.3361015319824219], [1.3358932733535767], [1.3356614112854004], [1.3354140520095825], [1.3351624011993408], [1.334914207458496], [1.3346784114837646], [1.3344610929489136], [1.3342684507369995], [1.334105372428894], [1.333974838256836], [1.3338791131973267], [1.3338180780410767], [1.333791971206665], [1.3337982892990112], [1.3338345289230347], [1.3338971138000488], [1.333982229232788], [1.3340845108032227], [1.334200382232666], [1.334323763847351], [1.3344504833221436], [1.3345760107040405], [1.3346956968307495], [1.3348063230514526], [1.3349052667617798], [1.334989070892334], [1.3350565433502197], [1.3351070880889893], [1.3351396322250366], [1.3351552486419678], [1.3351538181304932], [1.3351373672485352], [1.335107445716858], [1.3350658416748047], [1.3350152969360352], [1.3349578380584717], [1.334896206855774], [1.334832787513733], [1.3347694873809814], [1.3347088098526], [1.3346525430679321], [1.3346024751663208], [1.3345593214035034], [1.334524154663086], [1.3344975709915161], [1.3344802856445312], [1.334471583366394], [1.3344709873199463], [1.3344782590866089], [1.334492564201355], [1.3345125913619995], [1.3345378637313843], [1.3345661163330078], [1.334597110748291], [1.3346290588378906], [1.3346608877182007], [1.3346911668777466], [1.3347197771072388], [1.3347456455230713], [1.3347675800323486], [1.33478581905365], [1.334799885749817], [1.334808588027954], [1.3348138332366943], [1.3348143100738525], [1.3348113298416138], [1.3348045349121094], [1.3347947597503662], [1.3347828388214111], [1.334768533706665], [1.3347530364990234], [1.3347371816635132], [1.3347210884094238], [1.3347054719924927], [1.3346912860870361], [1.334678053855896], [1.3346666097640991], [1.3346573114395142], [1.3346501588821411], [1.33464515209198], [1.3346424102783203], [1.3346418142318726], [1.334643006324768], [1.3346463441848755], [1.3346513509750366], [1.334657073020935], [1.334663987159729], [1.3346717357635498], [1.3346794843673706], [1.3346877098083496], [1.33469557762146], [1.3347030878067017], [1.334709644317627], [1.334715723991394], [1.3347207307815552], [1.3347243070602417], [1.3347270488739014], [1.3347288370132446], [1.3347293138504028], [1.3347288370132446], [1.3347270488739014], [1.3347247838974], [1.334721326828003], [1.334718108177185], [1.3347140550613403], [1.3347100019454956], [1.3347057104110718], [1.3347018957138062], [1.3346980810165405], [1.3346946239471436], [1.3346920013427734], [1.3346890211105347], [1.3346872329711914], [1.3346858024597168], [1.334685206413269], [1.3346847295761108], [1.334685206413269], [1.3346855640411377], [1.3346868753433228], [1.3346885442733765], [1.3346906900405884], [1.3346925973892212], [1.3346946239471436], [1.3346964120864868], [1.3346985578536987], [1.3347002267837524], [1.3347020149230957], [1.3347035646438599], [1.334704875946045], [1.3347058296203613], [1.3347065448760986], [1.3347067832946777], [1.3347069025039673], [1.3347065448760986], [1.3347063064575195], [1.3347059488296509], [1.3347052335739136], [1.3347041606903076], [1.3347032070159912], [1.3347020149230957], [1.3347011804580688], [1.334700345993042], [1.334699273109436], [1.3346985578536987], [1.3346978425979614], [1.3346972465515137], [1.334696888923645], [1.334696650505066], [1.3346960544586182], [1.334695816040039], [1.3346959352493286], [1.3346961736679077], [1.334696650505066], [1.334696888923645], [1.3346970081329346], [1.3346974849700928], [1.33469820022583], [1.3346983194351196], [1.334699034690857]]
    


```python
# Transforma a saída

prev_optimal = scaler.inverse_transform(pred_output_optimal)
prev_optimal = np.array(prev_optimal).reshape(1, -1)
list_output_prev_optimal = list(prev_optimal)
list_output_prev_optimal = prev_optimal[0].tolist()
list_output_prev_optimal

# Pegar as data de previsão

dates_optimal = pd.to_datetime(historical_returns_optimized.index)
predict_dates_optimal = pd.date_range(list(dates_optimal)[-1] + pd.DateOffset(1), periods=n_future, freq = 'b').tolist()
predict_dates_optimal
```




    [Timestamp('2024-12-27 00:00:00'),
     Timestamp('2024-12-30 00:00:00'),
     Timestamp('2024-12-31 00:00:00'),
     Timestamp('2025-01-01 00:00:00'),
     Timestamp('2025-01-02 00:00:00'),
     Timestamp('2025-01-03 00:00:00'),
     Timestamp('2025-01-06 00:00:00'),
     Timestamp('2025-01-07 00:00:00'),
     Timestamp('2025-01-08 00:00:00'),
     Timestamp('2025-01-09 00:00:00'),
     Timestamp('2025-01-10 00:00:00'),
     Timestamp('2025-01-13 00:00:00'),
     Timestamp('2025-01-14 00:00:00'),
     Timestamp('2025-01-15 00:00:00'),
     Timestamp('2025-01-16 00:00:00'),
     Timestamp('2025-01-17 00:00:00'),
     Timestamp('2025-01-20 00:00:00'),
     Timestamp('2025-01-21 00:00:00'),
     Timestamp('2025-01-22 00:00:00'),
     Timestamp('2025-01-23 00:00:00'),
     Timestamp('2025-01-24 00:00:00'),
     Timestamp('2025-01-27 00:00:00'),
     Timestamp('2025-01-28 00:00:00'),
     Timestamp('2025-01-29 00:00:00'),
     Timestamp('2025-01-30 00:00:00'),
     Timestamp('2025-01-31 00:00:00'),
     Timestamp('2025-02-03 00:00:00'),
     Timestamp('2025-02-04 00:00:00'),
     Timestamp('2025-02-05 00:00:00'),
     Timestamp('2025-02-06 00:00:00'),
     Timestamp('2025-02-07 00:00:00'),
     Timestamp('2025-02-10 00:00:00'),
     Timestamp('2025-02-11 00:00:00'),
     Timestamp('2025-02-12 00:00:00'),
     Timestamp('2025-02-13 00:00:00'),
     Timestamp('2025-02-14 00:00:00'),
     Timestamp('2025-02-17 00:00:00'),
     Timestamp('2025-02-18 00:00:00'),
     Timestamp('2025-02-19 00:00:00'),
     Timestamp('2025-02-20 00:00:00'),
     Timestamp('2025-02-21 00:00:00'),
     Timestamp('2025-02-24 00:00:00'),
     Timestamp('2025-02-25 00:00:00'),
     Timestamp('2025-02-26 00:00:00'),
     Timestamp('2025-02-27 00:00:00'),
     Timestamp('2025-02-28 00:00:00'),
     Timestamp('2025-03-03 00:00:00'),
     Timestamp('2025-03-04 00:00:00'),
     Timestamp('2025-03-05 00:00:00'),
     Timestamp('2025-03-06 00:00:00'),
     Timestamp('2025-03-07 00:00:00'),
     Timestamp('2025-03-10 00:00:00'),
     Timestamp('2025-03-11 00:00:00'),
     Timestamp('2025-03-12 00:00:00'),
     Timestamp('2025-03-13 00:00:00'),
     Timestamp('2025-03-14 00:00:00'),
     Timestamp('2025-03-17 00:00:00'),
     Timestamp('2025-03-18 00:00:00'),
     Timestamp('2025-03-19 00:00:00'),
     Timestamp('2025-03-20 00:00:00'),
     Timestamp('2025-03-21 00:00:00'),
     Timestamp('2025-03-24 00:00:00'),
     Timestamp('2025-03-25 00:00:00'),
     Timestamp('2025-03-26 00:00:00'),
     Timestamp('2025-03-27 00:00:00'),
     Timestamp('2025-03-28 00:00:00'),
     Timestamp('2025-03-31 00:00:00'),
     Timestamp('2025-04-01 00:00:00'),
     Timestamp('2025-04-02 00:00:00'),
     Timestamp('2025-04-03 00:00:00'),
     Timestamp('2025-04-04 00:00:00'),
     Timestamp('2025-04-07 00:00:00'),
     Timestamp('2025-04-08 00:00:00'),
     Timestamp('2025-04-09 00:00:00'),
     Timestamp('2025-04-10 00:00:00'),
     Timestamp('2025-04-11 00:00:00'),
     Timestamp('2025-04-14 00:00:00'),
     Timestamp('2025-04-15 00:00:00'),
     Timestamp('2025-04-16 00:00:00'),
     Timestamp('2025-04-17 00:00:00'),
     Timestamp('2025-04-18 00:00:00'),
     Timestamp('2025-04-21 00:00:00'),
     Timestamp('2025-04-22 00:00:00'),
     Timestamp('2025-04-23 00:00:00'),
     Timestamp('2025-04-24 00:00:00'),
     Timestamp('2025-04-25 00:00:00'),
     Timestamp('2025-04-28 00:00:00'),
     Timestamp('2025-04-29 00:00:00'),
     Timestamp('2025-04-30 00:00:00'),
     Timestamp('2025-05-01 00:00:00'),
     Timestamp('2025-05-02 00:00:00'),
     Timestamp('2025-05-05 00:00:00'),
     Timestamp('2025-05-06 00:00:00'),
     Timestamp('2025-05-07 00:00:00'),
     Timestamp('2025-05-08 00:00:00'),
     Timestamp('2025-05-09 00:00:00'),
     Timestamp('2025-05-12 00:00:00'),
     Timestamp('2025-05-13 00:00:00'),
     Timestamp('2025-05-14 00:00:00'),
     Timestamp('2025-05-15 00:00:00'),
     Timestamp('2025-05-16 00:00:00'),
     Timestamp('2025-05-19 00:00:00'),
     Timestamp('2025-05-20 00:00:00'),
     Timestamp('2025-05-21 00:00:00'),
     Timestamp('2025-05-22 00:00:00'),
     Timestamp('2025-05-23 00:00:00'),
     Timestamp('2025-05-26 00:00:00'),
     Timestamp('2025-05-27 00:00:00'),
     Timestamp('2025-05-28 00:00:00'),
     Timestamp('2025-05-29 00:00:00'),
     Timestamp('2025-05-30 00:00:00'),
     Timestamp('2025-06-02 00:00:00'),
     Timestamp('2025-06-03 00:00:00'),
     Timestamp('2025-06-04 00:00:00'),
     Timestamp('2025-06-05 00:00:00'),
     Timestamp('2025-06-06 00:00:00'),
     Timestamp('2025-06-09 00:00:00'),
     Timestamp('2025-06-10 00:00:00'),
     Timestamp('2025-06-11 00:00:00'),
     Timestamp('2025-06-12 00:00:00'),
     Timestamp('2025-06-13 00:00:00'),
     Timestamp('2025-06-16 00:00:00'),
     Timestamp('2025-06-17 00:00:00'),
     Timestamp('2025-06-18 00:00:00'),
     Timestamp('2025-06-19 00:00:00'),
     Timestamp('2025-06-20 00:00:00'),
     Timestamp('2025-06-23 00:00:00'),
     Timestamp('2025-06-24 00:00:00'),
     Timestamp('2025-06-25 00:00:00'),
     Timestamp('2025-06-26 00:00:00'),
     Timestamp('2025-06-27 00:00:00'),
     Timestamp('2025-06-30 00:00:00'),
     Timestamp('2025-07-01 00:00:00'),
     Timestamp('2025-07-02 00:00:00'),
     Timestamp('2025-07-03 00:00:00'),
     Timestamp('2025-07-04 00:00:00'),
     Timestamp('2025-07-07 00:00:00'),
     Timestamp('2025-07-08 00:00:00'),
     Timestamp('2025-07-09 00:00:00'),
     Timestamp('2025-07-10 00:00:00'),
     Timestamp('2025-07-11 00:00:00'),
     Timestamp('2025-07-14 00:00:00'),
     Timestamp('2025-07-15 00:00:00'),
     Timestamp('2025-07-16 00:00:00'),
     Timestamp('2025-07-17 00:00:00'),
     Timestamp('2025-07-18 00:00:00'),
     Timestamp('2025-07-21 00:00:00'),
     Timestamp('2025-07-22 00:00:00'),
     Timestamp('2025-07-23 00:00:00'),
     Timestamp('2025-07-24 00:00:00'),
     Timestamp('2025-07-25 00:00:00'),
     Timestamp('2025-07-28 00:00:00'),
     Timestamp('2025-07-29 00:00:00'),
     Timestamp('2025-07-30 00:00:00'),
     Timestamp('2025-07-31 00:00:00'),
     Timestamp('2025-08-01 00:00:00'),
     Timestamp('2025-08-04 00:00:00'),
     Timestamp('2025-08-05 00:00:00'),
     Timestamp('2025-08-06 00:00:00'),
     Timestamp('2025-08-07 00:00:00'),
     Timestamp('2025-08-08 00:00:00'),
     Timestamp('2025-08-11 00:00:00'),
     Timestamp('2025-08-12 00:00:00'),
     Timestamp('2025-08-13 00:00:00'),
     Timestamp('2025-08-14 00:00:00'),
     Timestamp('2025-08-15 00:00:00'),
     Timestamp('2025-08-18 00:00:00'),
     Timestamp('2025-08-19 00:00:00'),
     Timestamp('2025-08-20 00:00:00'),
     Timestamp('2025-08-21 00:00:00'),
     Timestamp('2025-08-22 00:00:00'),
     Timestamp('2025-08-25 00:00:00'),
     Timestamp('2025-08-26 00:00:00'),
     Timestamp('2025-08-27 00:00:00'),
     Timestamp('2025-08-28 00:00:00'),
     Timestamp('2025-08-29 00:00:00'),
     Timestamp('2025-09-01 00:00:00'),
     Timestamp('2025-09-02 00:00:00'),
     Timestamp('2025-09-03 00:00:00'),
     Timestamp('2025-09-04 00:00:00'),
     Timestamp('2025-09-05 00:00:00'),
     Timestamp('2025-09-08 00:00:00'),
     Timestamp('2025-09-09 00:00:00'),
     Timestamp('2025-09-10 00:00:00'),
     Timestamp('2025-09-11 00:00:00'),
     Timestamp('2025-09-12 00:00:00'),
     Timestamp('2025-09-15 00:00:00'),
     Timestamp('2025-09-16 00:00:00'),
     Timestamp('2025-09-17 00:00:00'),
     Timestamp('2025-09-18 00:00:00'),
     Timestamp('2025-09-19 00:00:00'),
     Timestamp('2025-09-22 00:00:00'),
     Timestamp('2025-09-23 00:00:00'),
     Timestamp('2025-09-24 00:00:00'),
     Timestamp('2025-09-25 00:00:00'),
     Timestamp('2025-09-26 00:00:00'),
     Timestamp('2025-09-29 00:00:00'),
     Timestamp('2025-09-30 00:00:00'),
     Timestamp('2025-10-01 00:00:00'),
     Timestamp('2025-10-02 00:00:00'),
     Timestamp('2025-10-03 00:00:00'),
     Timestamp('2025-10-06 00:00:00'),
     Timestamp('2025-10-07 00:00:00'),
     Timestamp('2025-10-08 00:00:00'),
     Timestamp('2025-10-09 00:00:00'),
     Timestamp('2025-10-10 00:00:00'),
     Timestamp('2025-10-13 00:00:00'),
     Timestamp('2025-10-14 00:00:00'),
     Timestamp('2025-10-15 00:00:00'),
     Timestamp('2025-10-16 00:00:00'),
     Timestamp('2025-10-17 00:00:00'),
     Timestamp('2025-10-20 00:00:00'),
     Timestamp('2025-10-21 00:00:00'),
     Timestamp('2025-10-22 00:00:00'),
     Timestamp('2025-10-23 00:00:00'),
     Timestamp('2025-10-24 00:00:00'),
     Timestamp('2025-10-27 00:00:00'),
     Timestamp('2025-10-28 00:00:00'),
     Timestamp('2025-10-29 00:00:00'),
     Timestamp('2025-10-30 00:00:00'),
     Timestamp('2025-10-31 00:00:00'),
     Timestamp('2025-11-03 00:00:00'),
     Timestamp('2025-11-04 00:00:00'),
     Timestamp('2025-11-05 00:00:00'),
     Timestamp('2025-11-06 00:00:00'),
     Timestamp('2025-11-07 00:00:00'),
     Timestamp('2025-11-10 00:00:00'),
     Timestamp('2025-11-11 00:00:00'),
     Timestamp('2025-11-12 00:00:00'),
     Timestamp('2025-11-13 00:00:00'),
     Timestamp('2025-11-14 00:00:00'),
     Timestamp('2025-11-17 00:00:00'),
     Timestamp('2025-11-18 00:00:00'),
     Timestamp('2025-11-19 00:00:00'),
     Timestamp('2025-11-20 00:00:00'),
     Timestamp('2025-11-21 00:00:00'),
     Timestamp('2025-11-24 00:00:00'),
     Timestamp('2025-11-25 00:00:00'),
     Timestamp('2025-11-26 00:00:00'),
     Timestamp('2025-11-27 00:00:00'),
     Timestamp('2025-11-28 00:00:00'),
     Timestamp('2025-12-01 00:00:00'),
     Timestamp('2025-12-02 00:00:00'),
     Timestamp('2025-12-03 00:00:00'),
     Timestamp('2025-12-04 00:00:00'),
     Timestamp('2025-12-05 00:00:00'),
     Timestamp('2025-12-08 00:00:00'),
     Timestamp('2025-12-09 00:00:00'),
     Timestamp('2025-12-10 00:00:00'),
     Timestamp('2025-12-11 00:00:00'),
     Timestamp('2025-12-12 00:00:00'),
     Timestamp('2025-12-15 00:00:00')]




```python
# Cria o dataframe de previsão

forecast_dates_optimal = []

for i in predict_dates_optimal:
    forecast_dates_optimal.append(i.date())

df_forecast_optimal = pd.DataFrame({'Date':np.array(forecast_dates_optimal), 'Saldo':list_output_prev_optimal})
df_forecast_optimal['Date']=pd.to_datetime(df_forecast_optimal['Date'])

df_forecast_optimal = df_forecast_optimal.set_index(pd.DatetimeIndex(df_forecast_optimal['Date'].values))
df_forecast_optimal = df_forecast_optimal.drop('Date', axis=1)
df_forecast_optimal
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Saldo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-12-27</th>
      <td>190811.725434</td>
    </tr>
    <tr>
      <th>2024-12-30</th>
      <td>190959.305849</td>
    </tr>
    <tr>
      <th>2024-12-31</th>
      <td>190877.264571</td>
    </tr>
    <tr>
      <th>2025-01-01</th>
      <td>190860.776874</td>
    </tr>
    <tr>
      <th>2025-01-02</th>
      <td>190905.542091</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2025-12-09</th>
      <td>192560.295166</td>
    </tr>
    <tr>
      <th>2025-12-10</th>
      <td>192560.309610</td>
    </tr>
    <tr>
      <th>2025-12-11</th>
      <td>192560.331276</td>
    </tr>
    <tr>
      <th>2025-12-12</th>
      <td>192560.334887</td>
    </tr>
    <tr>
      <th>2025-12-15</th>
      <td>192560.356552</td>
    </tr>
  </tbody>
</table>
<p>252 rows × 1 columns</p>
</div>




```python

# Plota o gráfico da previsão
plt.figure(figsize=(12, 6))
plt.plot(portfolio_balance_optimal.tail(best_window), label='Histórico')
plt.plot(df_forecast_optimal, label='Previsão', color='red')
plt.xlabel('Data')
plt.ylabel('Saldo')
plt.title('Previsão do Saldo do Portfolio')
plt.legend()
plt.grid(True)

# Encontra e plota os valores máximo e mínimo da previsão
#max_value = df_forecast['Saldo'].max()
#min_value = df_forecast['Saldo'].min()
#max_date = df_forecast['Saldo'].idxmax()
#min_date = df_forecast['Saldo'].idxmin()

#plt.scatter(max_date, max_value, color='green', label='Máximo', s=100)
#plt.scatter(min_date, min_value, color='red', label='Mínimo', s=100)

#plt.annotate(f'Máximo: {max_value:.2f}', (max_date, max_value), xytext=(10,10),
             #textcoords='offset points', arrowprops=dict(arrowstyle='->'))
#plt.annotate(f'Mínimo: {min_value:.2f}', (min_date, min_value), xytext=(-80,-20),
             #textcoords='offset points', arrowprops=dict(arrowstyle='->'))

#plt.fill_between(df_forecast.index, df_forecast['Saldo'], min_value, color='orange', alpha=0.3)

plt.legend()
plt.show()

```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_157_0.png)
    


# 7 - Comparação de Investimentos

## VaR , Backtesting e Stress Test


```python
###### VaR - Value at Risk ######

## Portfolio atual

print('Janelas do VaR para o Portfólio Escolhido')

# Mostrar todos os resultados armazenados em test_windows_results
print("\nResultados do VaR para diferentes janelas de tempo:")
for window, var in test_windows_results:
    print(f"Janela de {window} dias: VaR = R$ {var:.2f}")

# Encontrar o melhor valor de VaR (o menor, pois ele representa a perda máxima)
best_window, best_VaR = min(test_windows_results, key=lambda x: x[1])
print(f'\nMelhor janela de tempo: {best_window} dias com VaR = R$ {best_VaR:.2f}')

# Imprimir a taxa de violação observada
taxa_violacao = np.mean(violacoes)
print(f'Taxa de Violação Observada: {taxa_violacao:.4f}')
# Exibir o resultado do VaR em cenário de stress
print(f'VaR em cenário de stress (choque de {shock_factor*100}%): R$ {VaR_stress:.2f}\n')


print('--------------------------------------------------------------')

## Portfólio Otimizado

print('\nJanelas do VaR para o Portfólio Otimizado')

# Mostrar todos os resultados armazenados em test_windows_results
print("\nResultados do VaR para diferentes janelas de tempo:")
for window, var in test_windows_results_optimal:
    print(f"Janela de {window} dias: VaR = R$ {var:.2f}")
# Encontrar o melhor valor de VaR (o menor, pois ele representa a perda máxima)
best_window_optimal, best_VaR_optimal = min(test_windows_results_optimal, key=lambda x: x[1])
print(f'\nMelhor janela de tempo: {best_window_optimal} dias com VaR = R$ {best_VaR_optimal:.2f}')
# Imprimir a taxa de violação observada
taxa_violacao_port_optimized = np.mean(violacoes_port_optimized)
print(f'Taxa de Violação do Portfólio Ótimo Observada: {taxa_violacao_port_optimized:.4f}')
# Exibir o resultado do VaR em cenário de stress
print(f'VaR em cenário de stress (choque de {shock_factor*100}%): R$ {VaR_stress_optimal:.2f}\n\n')
```

    Janelas do VaR para o Portfólio Escolhido
    
    Resultados do VaR para diferentes janelas de tempo:
    Janela de 30 dias: VaR = R$ 8564.20
    Janela de 60 dias: VaR = R$ 9009.10
    Janela de 90 dias: VaR = R$ 7769.46
    Janela de 180 dias: VaR = R$ 846.81
    Janela de 252 dias: VaR = R$ -540.24
    
    Melhor janela de tempo: 252 dias com VaR = R$ -540.24
    Taxa de Violação Observada: 0.9518
    VaR em cenário de stress (choque de 50.0%): R$ -270.12
    
    --------------------------------------------------------------
    
    Janelas do VaR para o Portfólio Otimizado
    
    Resultados do VaR para diferentes janelas de tempo:
    Janela de 30 dias: VaR = R$ 8970.89
    Janela de 60 dias: VaR = R$ 8310.51
    Janela de 90 dias: VaR = R$ 8532.67
    Janela de 180 dias: VaR = R$ -3562.10
    Janela de 252 dias: VaR = R$ -11953.32
    
    Melhor janela de tempo: 252 dias com VaR = R$ -11953.32
    Taxa de Violação do Portfólio Ótimo Observada: 1.0000
    VaR em cenário de stress (choque de 50.0%): R$ -5976.66
    
    
    


```python
# Configurar o layout 2x2
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Plotar o primeiro gráfico
range_returns_best = historical_returns.rolling(window=best_window).sum().dropna()
axs[0, 0].hist(range_returns_best * portfolio_value, bins=50, density=True, alpha=0.6, color='g')
axs[0, 0].axvline(-best_VaR, color='r', linestyle='dashed', linewidth=2, label=f'VaR {confidence_interval*100}% de confiança')
axs[0, 0].set_xlabel(f'{best_window} dias - Retorno do Portfolio (Reais)')
axs[0, 0].set_ylabel('Frequência')
axs[0, 0].set_title(f'Distribuição dos Retornos do Portfolio - {best_window} dias (Reais)')
axs[0, 0].legend()

# Plotar o segundo gráfico
axs[0, 1].hist(range_returns_stress_optimal * portfolio_value, bins=50, density=True, alpha=0.6, color='r')
axs[0, 1].axvline(-VaR_stress_optimal, color='b', linestyle='dashed', linewidth=2, label=f'VaR Stress {confidence_interval*100}% de confiança')
axs[0, 1].set_xlabel(f'{best_window_optimal} dias - Retorno do Portfolio (Reais) em Stress')
axs[0, 1].set_ylabel('Frequência')
axs[0, 1].set_title(f'Distribuição dos Retornos do Portfolio - {best_window_optimal} dias (Stress)')
axs[0, 1].legend()

# Plotar o terceiro gráfico
range_returns_best_optimized = historical_returns_optimized.rolling(window=best_window_optimal).sum().dropna()
axs[1, 0].hist(range_returns_best_optimized * portfolio_value, bins=50, density=True, alpha=0.6, color='g')
axs[1, 0].axvline(-best_VaR_optimal, color='r', linestyle='dashed', linewidth=2, label=f'VaR {confidence_interval*100}% de confiança')
axs[1, 0].set_xlabel(f'{best_window_optimal} dias - Retorno do Portfolio (Reais)')
axs[1, 0].set_ylabel('Frequência')
axs[1, 0].set_title(f'Distribuição dos Retornos do Portfolio - {best_window_optimal} dias (Reais)')
axs[1, 0].legend()

# Plotar o quarto gráfico
axs[1, 1].hist(range_returns_stress_optimal * portfolio_value, bins=50, density=True, alpha=0.6, color='r')
axs[1, 1].axvline(-VaR_stress_optimal, color='b', linestyle='dashed', linewidth=2, label=f'VaR Stress {confidence_interval*100}% de confiança')
axs[1, 1].set_xlabel(f'{best_window_optimal} dias - Retorno do Portfolio (Reais) em Stress')
axs[1, 1].set_ylabel('Frequência')
axs[1, 1].set_title(f'Distribuição dos Retornos do Portfolio - {best_window_optimal} dias (Stress)')
axs[1, 1].legend()

# Ajustar o layout para não sobrepor os gráficos
plt.tight_layout()
plt.show()
```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_161_0.png)
    


## Análise dos Retornos


```python

# Imprimir os resultados
print(f"Retornos do Portfolio Atual\n")

print(f"Saldo atual do Portfolio: R$ {saldo_atual_portfolio:.2f}")
print(f"Saldo atual do CDI: R$ {saldo_atual_cdi:.2f}")
print(f"Saldo atual do IBOV: R$ {saldo_atual_ibov:.2f}\n")
print(f"Diferença Portfolio vs. CDI: R$ {diferenca_cdi:.2f}")
print(f"Diferença Portfolio vs. IBOV: R$ {diferenca_ibov:.2f}\n")

print('--------------------------------------------------------------')

# Imprimir os resultados
print(f"Retornos do Portfolio Otimizado\n")

print(f"Saldo atual do Portfolio Otimizado: R$ {saldo_atual_portfolio_optimal:.2f}")
print(f"Saldo atual do CDI: R$ {saldo_atual_cdi:.2f}")
print(f"Saldo atual do IBOV: R$ {saldo_atual_ibov:.2f}\n")
print(f"Diferença Portfolio Otimizado vs. CDI: R$ {diferenca_cdi_optimal:.2f}")
print(f"Diferença Portfolio Otimizado vs. IBOV: R$ {diferenca_ibov_optimal:.2f}\n")
```

    Retornos do Portfolio Atual
    
    Saldo atual do Portfolio: R$ 140650.63
    Saldo atual do CDI: R$ 134261.80
    Saldo atual do IBOV: R$ 114376.66
    
    Diferença Portfolio vs. CDI: R$ 6388.83
    Diferença Portfolio vs. IBOV: R$ 26273.97
    
    --------------------------------------------------------------
    Retornos do Portfolio Otimizado
    
    Saldo atual do Portfolio Otimizado: R$ 192006.19
    Saldo atual do CDI: R$ 134261.80
    Saldo atual do IBOV: R$ 114376.66
    
    Diferença Portfolio Otimizado vs. CDI: R$ 57744.39
    Diferença Portfolio Otimizado vs. IBOV: R$ 77629.54
    
    


```python
# Plota os retornos acumulados
plt.figure(figsize=(12, 6))
plt.plot(historical_returns.index, portfolio_return, label='Portfolio')
plt.plot(historical_returns_optimized.index, portfolio_return_optimized, label='Portfolio Otimizado')
plt.plot(cdi.index, cdi_return, label='CDI', linestyle='--')
plt.plot(log_ibov.index, ibov_return, label='IBOV', linestyle='-.')
plt.axhline(portfolio_value, color='k', linestyle=':', label='Investimento Inicial')
plt.title('Retorno Acumulado do Portfolio vs. Otimizado vs. Benchmarks (R$)')
plt.xlabel('Data')
plt.ylabel('Valor Acumulado (R$)')
plt.legend()
plt.grid(True)
plt.show()




```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_164_0.png)
    


## CAPM 


```python
##### CAPM  #####

## Portfolio atual

# Imprime os resultados
print(f"Análise CAPM do Portfolio Atual\n")
print(f"Beta do Portfolio: {beta_portfolio:.4f}")
print(f"Retorno Esperado (CAPM): {expected_return_capm:.4f}")
print(f"Alfa do Portfolio: {alpha_portfolio:.4f}")

# Imprime os resultados adicionais
print(f"Erro de Previsão do CAPM: {capm_prediction_error:.4f}")
print(f"R-quadrado do Modelo CAPM: {r_squared:.4f}")

# Interpretação dos resultados
print("\nInterpretação dos Resultados:")
print(f"- Beta do Portfolio ({beta_portfolio:.4f}): Indica que o portfolio é {beta_portfolio:.2f} vezes mais volátil que o IBOV.")
if alpha_portfolio > 0:
    print(f"- Alfa do Portfolio ({alpha_portfolio:.4f}): Positivo, indicando que o portfolio gerou retornos acima do esperado pelo CAPM, sugerindo uma possível habilidade do gestor.")
elif alpha_portfolio < 0:
    print(f"- Alfa do Portfolio ({alpha_portfolio:.4f}): Negativo, indicando que o portfolio gerou retornos abaixo do esperado pelo CAPM.")
else:
    print(f"- Alfa do Portfolio ({alpha_portfolio:.4f}): Zero, indicando que o portfolio gerou retornos em linha com o esperado pelo CAPM.")
print(f"- R-quadrado do Modelo CAPM ({r_squared:.4f}): Indica que {r_squared*100:.2f}% da variância do retorno do portfolio é explicada pelo IBOV.")

print('--------------------------------------------------------------')

## Portfólio Otimizado


# Imprime os resultados
print(f"Análise CAPM do Portfolio Otimizado\n")

print(f"Beta do Portfolio: {beta_portfolio_optimal:.4f}")
print(f"Retorno Esperado (CAPM): {expected_return_capm_optimal:.4f}")
print(f"Alfa do Portfolio: {alpha_portfolio_optimal:.4f}")

# Imprime os resultados adicionais
print(f"Erro de Previsão do CAPM: {capm_prediction_error_optimal:.4f}")
print(f"R-quadrado do Modelo CAPM: {r_squared_capm_optimal:.4f}")

# Interpretação dos resultados
print("\nInterpretação dos Resultados:")
print(f"- Beta do Portfolio ({beta_portfolio_optimal:.4f}): Indica que o portfolio ótimo é {beta_portfolio_optimal:.2f} vezes mais volátil que o IBOV.")
if alpha_portfolio > 0:
    print(f"- Alfa do Portfolio Ótimo ({alpha_portfolio_optimal:.4f}): Positivo, indicando que o portfolio ótimo gerou retornos acima do esperado pelo CAPM, sugerindo uma possível habilidade do gestor.")
elif alpha_portfolio < 0:
    print(f"- Alfa do Portfolio Ótimo ({alpha_portfolio_optimal:.4f}): Negativo, indicando que o portfolio ótimo gerou retornos abaixo do esperado pelo CAPM.")
else:
    print(f"- Alfa do Portfolio Ótimo ({alpha_portfolio_optimal:.4f}): Zero, indicando que o portfolio ótimo gerou retornos em linha com o esperado pelo CAPM.")
print(f"- R-quadrado do Modelo CAPM ({r_squared_capm_optimal:.4f}): Indica que {r_squared_capm_optimal*100:.2f}% da variância do retorno do portfolio ótimo é explicada pelo IBOV.")
```

    Análise CAPM do Portfolio Atual
    
    Beta do Portfolio: 1.0140
    Retorno Esperado (CAPM): 0.0490
    Alfa do Portfolio: 0.0877
    Erro de Previsão do CAPM: 0.0877
    R-quadrado do Modelo CAPM: 0.8421
    
    Interpretação dos Resultados:
    - Beta do Portfolio (1.0140): Indica que o portfolio é 1.01 vezes mais volátil que o IBOV.
    - Alfa do Portfolio (0.0877): Positivo, indicando que o portfolio gerou retornos acima do esperado pelo CAPM, sugerindo uma possível habilidade do gestor.
    - R-quadrado do Modelo CAPM (0.8421): Indica que 84.21% da variância do retorno do portfolio é explicada pelo IBOV.
    --------------------------------------------------------------
    Análise CAPM do Portfolio Otimizado
    
    Beta do Portfolio: 0.9525
    Retorno Esperado (CAPM): 0.0461
    Alfa do Portfolio: 0.2635
    Erro de Previsão do CAPM: 0.2635
    R-quadrado do Modelo CAPM: 0.5661
    
    Interpretação dos Resultados:
    - Beta do Portfolio (0.9525): Indica que o portfolio ótimo é 0.95 vezes mais volátil que o IBOV.
    - Alfa do Portfolio Ótimo (0.2635): Positivo, indicando que o portfolio ótimo gerou retornos acima do esperado pelo CAPM, sugerindo uma possível habilidade do gestor.
    - R-quadrado do Modelo CAPM (0.5661): Indica que 56.61% da variância do retorno do portfolio ótimo é explicada pelo IBOV.
    


```python


# Configurar o layout 1x2 para colocar os gráficos lado a lado
fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# Primeiro gráfico: Relação entre Retorno do Portfolio e Retorno do IBOV
axs[0].scatter(log_ibov['IBOV'], historical_returns, alpha=0.6)
axs[0].set_xlabel('Retorno do IBOV')
axs[0].set_ylabel('Retorno do Portfolio')
axs[0].set_title('Relação entre Retorno do Portfolio e Retorno do IBOV')

# Adicionar a linha de regressão (CAPM)
x = np.linspace(log_ibov['IBOV'].min(), log_ibov['IBOV'].max(), 100)
y = risk_free_rate + beta_portfolio * (x - risk_free_rate)
axs[0].plot(x, y, color='red', label='Linha de Regressão (CAPM)')

axs[0].legend()
axs[0].grid(True)

# Segundo gráfico: Relação entre Retorno do Portfolio Ótimo e Retorno do IBOV
axs[1].scatter(log_ibov['IBOV'], historical_returns_optimized, alpha=0.6)
axs[1].set_xlabel('Retorno do IBOV')
axs[1].set_ylabel('Retorno do Portfolio Ótimo')
axs[1].set_title('Relação entre Retorno do Portfolio Ótimo e Retorno do IBOV')

# Adicionar a linha de regressão (CAPM)
y = risk_free_rate + beta_portfolio_optimal * (x - risk_free_rate)
axs[1].plot(x, y, color='red', label='Linha de Regressão (CAPM)')

axs[1].legend()
axs[1].grid(True)

# Ajustar o layout para não sobrepor os gráficos
plt.tight_layout()
plt.show()

```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_167_0.png)
    


## Regresão Linear


```python
# Plotando os resultados
plt.figure(figsize=(14, 8))

# Histórico
for coluna, cor in zip(['returns'], [ 'green']):
    plt.plot(df_modelagem_otima.index, saldos_optimal['historico'][coluna], label=f'Saldo Histórico ({coluna.upper()})', color=cor)

# Histórico
for coluna, cor in zip(['returns', 'IBOV', 'cdi'], ['blue', 'green', 'orange']):
    plt.plot(df_modelagem.index, saldos['historico'][coluna], label=f'Saldo Histórico ({coluna.upper()})', color=cor)

# Futuro
for coluna, cor in zip(['returns', 'IBOV', 'cdi'], ['blue', 'green', 'orange']):
    plt.plot(resultados[coluna]['datas_futuras'], saldos['futuro'][coluna], linestyle='--', label=f'Saldo Previsto ({coluna.upper()})', color=cor)

# Futuro
for coluna, cor in zip(['returns'], ['green']):
    plt.plot(resultados[coluna]['datas_futuras'], saldos_optimal['futuro'][coluna], linestyle='--', label=f'Saldo Previsto ({coluna.upper()})', color=cor)

# Linha do investimento inicial
plt.axhline(y=valor_investido, color='red', linestyle='--', label='Saldo Inicial (Investimento)')

# Configurações do gráfico
plt.xlabel('Data')
plt.ylabel('Saldo (R$)')
plt.title('Comparação de Saldo: Retornos, IBOV e CDI (Histórico e Previsão)')
plt.legend()
plt.grid(True)
plt.show()

# Métricas da regressão otima
for coluna in ['returns']:
    print(f"\nMétricas para {coluna.upper()} ótima:")
    print(f"  - MSE: {resultados_optimal[coluna]['mse']:.6f}")
    print(f"  - MAE: {resultados_optimal[coluna]['mae']:.6f}")
    print(f"  - R²: {resultados_optimal[coluna]['r2']:.6f}")

# Métricas da regressão
for coluna in ['returns', 'IBOV', 'cdi']:
    print(f"\nMétricas para {coluna.upper()}:")
    print(f"  - MSE: {resultados[coluna]['mse']:.6f}")
    print(f"  - MAE: {resultados[coluna]['mae']:.6f}")
    print(f"  - R²: {resultados[coluna]['r2']:.6f}")
```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_169_0.png)
    


    
    Métricas para RETURNS ótima:
      - MSE: 0.000192
      - MAE: 0.010315
      - R²: 0.000437
    
    Métricas para RETURNS:
      - MSE: 0.000146
      - MAE: 0.009270
      - R²: 0.001720
    
    Métricas para IBOV:
      - MSE: 0.000120
      - MAE: 0.008454
      - R²: 0.000258
    
    Métricas para CDI:
      - MSE: 0.000000
      - MAE: 0.000038
      - R²: 0.147756
    

## ARIMA


```python
plt.figure(figsize=(14, 7))

plt.plot(portfolio_return.index, portfolio_return, label='Histórico Portfólio', color='blue')
plt.plot(forecast_index, forecast_mean, label='Previsão Portfólio', color='blue', linestyle='--')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='lightsteelblue', alpha=0.3, label='Intervalo de Confiança Portfólio')

plt.plot(portfolio_return_optimized.index, portfolio_return_optimized, label='Histórico Portfolio Ótimo', color='cyan')
plt.plot(forecast_index, forecast_mean_optimal, label='Previsão Portfólio Ótimo', color='aqua',  linestyle='--')
plt.fill_between(forecast_index, forecast_conf_int_optimal.iloc[:, 0], forecast_conf_int_optimal.iloc[:, 1], color='darkturquoise', alpha=0.3, label='Intervalo de Confiança Port. Ótimo')

plt.plot(cdi_return.index, cdi_return, label='Histórico CDI', color='green')
plt.plot(forecast_index_cdi, forecast_mean_cdi, label='Previsão CDI', color='green',  linestyle='--')
plt.fill_between(forecast_index_cdi, forecast_conf_int_cdi.iloc[:, 0], forecast_conf_int_cdi.iloc[:, 1], color='darkseagreen', alpha=0.3, label='Intervalo de Confiança CDI')

plt.plot(ibov_return.index, ibov_return, label='Histórico IBOV', color='orange')
plt.plot(forecast_index_ibov, forecast_mean_ibov, label='Previsão IBOV', color='orange',  linestyle='--')
plt.fill_between(forecast_index_ibov, forecast_conf_int_ibov.iloc[:, 0], forecast_conf_int_ibov.iloc[:, 1], color='goldenrod', alpha=0.3, label='Intervalo de Confiança IBOV')

plt.xlabel('Data')
plt.ylabel('Retorno')
plt.title(f'Comparação entre Histórico e Previsão para Portfólio, CDI e IBOV ({best_window} dias)')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_171_0.png)
    


## LSTM


```python
# Plota o gráfico da previsão
plt.figure(figsize=(12, 6))
plt.plot(portfolio_balance.tail(best_window), label='Histórico do Portfólio', color='blue')
plt.plot(df_forecast, label='Previsão do Portfólio', color='red')
plt.plot(portfolio_balance_optimal.tail(best_window), label='Histórico do Portfólio Ótimo', color='green')
plt.plot(df_forecast_optimal, label='Previsão do Portfólio Ótimo', color='red')
plt.xlabel('Data')
plt.ylabel('Saldo')
plt.title('Previsão do Saldo do Portfolio')
plt.legend()
plt.grid(True)
plt.legend()
plt.show()

```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_173_0.png)
    



```python
plt.figure(figsize=(14, 7))

plt.plot(portfolio_return.index, portfolio_return, label='Histórico Portfólio', color='blue')
plt.plot(forecast_index, forecast_mean, label='Previsão Portfólio', color='blue', linestyle='--')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='lightsteelblue', alpha=0.3, label='Intervalo de Confiança Portfólio')

plt.plot(portfolio_return_optimized.index, portfolio_return_optimized, label='Histórico Portfolio Ótimo', color='cyan')
plt.plot(forecast_index, forecast_mean_optimal, label='Previsão Portfólio Ótimo', color='aqua',  linestyle='--')
plt.fill_between(forecast_index, forecast_conf_int_optimal.iloc[:, 0], forecast_conf_int_optimal.iloc[:, 1], color='darkturquoise', alpha=0.3, label='Intervalo de Confiança Port. Ótimo')

plt.plot(cdi_return.index, cdi_return, label='Histórico CDI', color='green')
plt.plot(forecast_index_cdi, forecast_mean_cdi, label='Previsão CDI', color='green',  linestyle='--')
plt.fill_between(forecast_index_cdi, forecast_conf_int_cdi.iloc[:, 0], forecast_conf_int_cdi.iloc[:, 1], color='darkseagreen', alpha=0.3, label='Intervalo de Confiança CDI')

plt.plot(ibov_return.index, ibov_return, label='Histórico IBOV', color='orange')
plt.plot(forecast_index_ibov, forecast_mean_ibov, label='Previsão IBOV', color='orange',  linestyle='--')
plt.fill_between(forecast_index_ibov, forecast_conf_int_ibov.iloc[:, 0], forecast_conf_int_ibov.iloc[:, 1], color='goldenrod', alpha=0.3, label='Intervalo de Confiança IBOV')

plt.plot(df_forecast, label='Previsão do Portfólio', color='red')
plt.plot(df_forecast_optimal, label='Previsão do Portfólio Ótimo', color='purple')

plt.xlabel('Data')
plt.ylabel('Retorno')
plt.title(f'Comparação entre Histórico e Previsão para Portfólio, CDI e IBOV ({best_window} dias)')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_174_0.png)
    



```python


# Plota o gráfico da previsão
plt.figure(figsize=(12, 6))
plt.plot(portfolio_balance.tail(252), label='Histórico do Portfólio', color='blue')
plt.plot(df_forecast, label='Previsão do Portfólio (LSTM)', color='red')
plt.plot(forecast_index, forecast_mean, label='Previsão Portfólio (ARIMA)', color='cyan')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Intervalo de Confiança Portfólio (ARIMA)')

# Linha do investimento inicial
plt.axhline(y=valor_investido, color='red', linestyle='--', label='Saldo Inicial (Investimento)')

plt.plot(portfolio_balance_optimal.tail(252), label='Histórico do Portfólio Ótimo', color='green')
plt.plot(df_forecast_optimal, label='Previsão do Portfólio Ótimo (LSTM)', color='orange')
plt.plot(forecast_index, forecast_mean_optimal, label='Previsão Portfólio Ótimo (ARIMA)', color='olive')
plt.fill_between(forecast_index, forecast_conf_int_optimal.iloc[:, 0], forecast_conf_int_optimal.iloc[:, 1], color='gray', alpha=0.3, label='Intervalo de Confiança Portfólio Ótimo (ARIMA)')
plt.xlabel('Data')
plt.xlabel('Data')
plt.ylabel('Saldo')
plt.title('Previsão do Saldo do Portfolio')
plt.legend()
plt.grid(True)
plt.legend()
plt.show()
```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_175_0.png)
    



```python
# Definindo o valor inicial do investimento
initial_investment = 100000

# Cálculo dos retornos previstos para os dois portfólios (comum e ótimo)
# Vamos assumir que df_forecast e df_forecast_optimal possuem os retornos diários previstos

# Calculando o retorno diário previsto e acumulado para o portfólio comum
portfolio_returns = df_forecast.pct_change().fillna(0) + 1  # Retornos previstos do portfólio
portfolio_optimal_returns = df_forecast_optimal.pct_change().fillna(0) + 1  # Retornos previstos do portfólio ótimo

# Calculando a evolução do valor do investimento ao longo do tempo
portfolio_value = initial_investment * portfolio_returns.cumprod()  # Valor do portfólio comum
portfolio_optimal_value = initial_investment * portfolio_optimal_returns.cumprod()  # Valor do portfólio ótimo

# Plotando o gráfico com as trajetórias de investimento
plt.figure(figsize=(12, 6))

# Trajetória de investimento para o portfólio comum
plt.plot(portfolio_value, label='Investimento no Portfólio Comum', color='blue', linewidth=2)

# Trajetória de investimento para o portfólio ótimo
plt.plot(portfolio_optimal_value, label='Investimento no Portfólio Ótimo', color='cyan', linewidth=2)

# Adicionando rótulos e título
plt.xlabel('Data', fontsize=14)
plt.ylabel('Valor do Investimento (R$)', fontsize=14)
plt.title('Trajetória do Investimento Inicial de R$ 100.000,00', fontsize=16)

# Legenda
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

# Grid
plt.grid(True, linestyle='--', alpha=0.5)

# Ajustando layout
plt.tight_layout()

# Mostrar o gráfico
plt.show()

```


    
![png](TCC_Finances%20-%20Copia_files/TCC_Finances%20-%20Copia_176_0.png)
    


# 8 - Conclusão

Este trabalho visa oferecer uma ferramenta prática e eficiente para investidores, utilizando técnicas de Machine Learning para analisar e comparar diferentes opções de investimento. Através da previsão de retornos e análise de risco, espera-se proporcionar insights valiosos que auxiliem na tomada de decisões financeiras mais informadas e estratégicas.
