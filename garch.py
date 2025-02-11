import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import chi2
from pmdarima.model_selection import train_test_split
from statsmodels.stats.diagnostic import het_arch

from arch import arch_model

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


### Turns out the following implementation is wrong so commenting it out for now
# def McLeodLi_test(x, k):
#     n = len(x)
#     x_sq = x ** 2
#     x_sum = np.sum(x_sq)
#     x_lag_sum = np.sum(x_sq[:-k])              
#     test = n * (n + 2) * x_lag_sum / (x_sum ** 2)
#     # df = k
#     p_value = 1 - chi2.cdf(test, k)
#     return test, p_value

### Deepseek and Claude verified McLeod-li test implementation
def McLeodLi_test(x, k):
    x = np.asarray(x)
    n = len(x)
    if k < 1 or k >= n:
        raise ValueError("k must be between 1 and n-1 inclusive")
    x_sq = x ** 2
    test_stat = 0.0
    for i in range(1, k + 1):
        if n - i <= 0:
            raise ValueError(f"Invalid lag i={i} for n={n}")
        r_i = np.corrcoef(x_sq[i:], x_sq[:-i])[0, 1]
        test_stat += r_i ** 2 / (n - i)
    test_stat *= n * (n + 2)
    p_value = 1 - chi2.cdf(test_stat, k)
    return test_stat, p_value



# spy = yf.Ticker("SPY")
spy = yf.Ticker("^NSEI")
hist = spy.history(start="2010-01-04", end="2020-02-01")
df = pd.DataFrame(hist, columns=['Close'])
print(df.head)

df['Return'] = np.pad(np.diff(np.log(df['Close'])) * 100, (1, 0), 'constant', constant_values=np.nan)       # log returns (additive property thru log)

# print(df.head)

plt.figure(figsize=(8,5))
plt.plot(df['Return'])
plt.ylabel('Return %')
plt.title('Return volatility')
# plt.show()

ts_returns = df['Return'].iloc[1:]              # Timeseries Returns without first NaN value

### Commenting below graphs
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
# plot_acf(ts_returns, ax=ax1, lags=10)
# ax1.set_ylim(-0.5, 0.5)
# ax1.set_title("Autocorrelation (ACF)")
# plot_pacf(ts_returns, ax=ax2, lags=10)
# ax2.set_ylim(-0.5, 0.5)
# ax2.set_xlabel("Lag")
# ax2.set_title("Partial Autocorrelation (PACF)")

# plt.show()


abs_returns = ts_returns.abs()                              # absolute values, convert all - to +
# print(abs_returns)

####### Commenting below graphs
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
# plot_acf(abs_returns, ax=ax1, lags=10)
# ax1.set_ylim(-0.5, .5)
# ax1.set_title("ACF for Absolute Returns")
# plot_pacf(abs_returns, ax=ax2, lags=10)
# ax2.set_ylim(-0.5, 0.5)
# ax2.set_title("PACF for Asbolute Returns")
# ax2.set_xlabel("Lag")

# plt.show()

# CAPTURES THE ARCH effect on the base dataset
test, p_value = McLeodLi_test(abs_returns, 50)
print("----------------MCLT on base data----------------")
print("McLeod-Li Test Statistics: ", test)
print("p-Value: ", p_value)

y_train, y_test = train_test_split(abs_returns, train_size=0.8)

# print(f"Returns: {np.mean(abs_returns)}")

# garch = arch_model(y_train, vol="Garch", p=1, q=1, rescale=False)#, mean="Zero"           # normal GARCH
garch = arch_model(y_train, vol="Garch", p=1, q=1, o=1, rescale=False)#, mean="Zero")                   # GJR GARCH
res_garch = garch.fit()

print(res_garch.summary())
print(f"AIC: {res_garch.aic} and BIC: {res_garch.bic}")

resid = res_garch.resid
# print("--------------RESIDUALS------------")
# print(resid)
lag_order = 10
arch_test = het_arch(resid, nlags=lag_order)
# print("Arch Test: ", arch_test)
print(f"ARCH-LM p-value: {arch_test[1]}")               # #
gamma_value, gamma_pvalue = res_garch.params['gamma[1]'], res_garch.pvalues['gamma[1]']
gamma_p_value_bool = gamma_pvalue < 0.05
print("Gamma Significant (p < 0.05): ", gamma_p_value_bool)                             # signifies gamma is statistically significant and asymmetric volatility exists


# print("---------------conditional volatility---------------")
# print(res_garch.conditional_volatility)
standardised_residuals = res_garch.resid / res_garch.conditional_volatility
test_stat, p_value = McLeodLi_test(standardised_residuals, 50)
print("McLeod-Li Test Statistic (after GARCH): ", test_stat)
print("p-Value: ", p_value)


# print(y_test.shape[0])          # 508
# print(y_test.index)
yhat = res_garch.forecast(horizon=y_test.shape[0], reindex=True)
# print(type(yhat.variance.values))
# print(np.sqrt(yhat.variance.values[-1]))

# sqrt(0.3612691) = 0.60105739


##### Below is n-step ahead forecast (basically just t+1 and t+n depends on t (not (t+n)-1 like it ideally should))

# fig, ax = plt.subplots(figsize=(10,8))
# ax.spines[['top', 'right']].set_visible(False)
# plt.plot(ts_returns[-y_test.shape[0]:])                 # plot onlt test split
# plt.plot(y_test.index, np.sqrt(yhat.variance.values[-1,:]))                            # plot volatility estimate; y_test.index = dates;
# plt.title('Volatility')
# plt.legend(['Daily log returns', 'predicted volatility'])
# # plt.show()
# res_garch.plot(annualize="D")
# plt.show()


##### One Step ahead rolling forecast

rolling_preds = []
for i in range(y_test.shape[0]):
    train = abs_returns[:-(y_test.shape[0]-i)]              # want next time period return (t+1) from abs_returns # list [] increases with new value with each passing period (in loop i)
    model = arch_model(train, p=1, q=1, o=1, rescale=False)     # changed GARCH to GJR GARCH
    model_fit = model.fit(disp='off')
    # one step ahead prediction
    predict = model_fit.forecast(horizon=1, reindex=True)
    # print(predict.variance.values)
    rolling_preds.append(np.sqrt(predict.variance.values[-1,:][0]))

rolling_preds = pd.Series(rolling_preds, index=y_test.index)            #

# print(rolling_preds)

#### Single graph of new predictions from one step rolling forecast
# fig, ax = plt.subplots(figsize=(10,4))
# ax.spines[['top', 'right']].set_visible(False)                  # used to hide top and right border lines on the graph
# plt.plot(rolling_preds)
# plt.title('S&P 500 Rolling Volatility Prediction')
# plt.show()



# commenting the graph plot
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
# ax1.spines[['top', 'right']].set_visible(False)
# ax1.plot(ts_returns[-y_test.shape[0]:])
# ax1.plot(y_test.index, np.sqrt(yhat.variance.values[-1]))
# ax1.set_title("N Step Predictions")
# ax1.legend(['True Daily Returns', 'Volatility Predictions'])
#
# ax2.spines[['top', 'right']].set_visible(False)
# ax2.plot(ts_returns[-y_test.shape[0]:])                             # plots everything after size of y_test (plots fill y_test)
# ax2.plot(rolling_preds)
# ax2.set_title("One Step Rolling Predictions")
# ax2.legend(['True Daily Returns', 'Volatility Predictions'])
# plt.show()




################### GARCH(4,4) has worse/higher AIC and BIC than GARCH(1,1)


############# ADDING XGBoost to the process ###############################

conditional_volatility = res_garch.conditional_volatility
residuals = res_garch.resid

# print("-----------------------")
# print(conditional_volatility)
# print(residuals)
# print("xxxxxxxxxxxxxx")

# vix = yf.download("^VIX", start=df.index.min(), end=df.index.max())['Close'].shift(1).rename('VIX')
print("-----------OLD VIX--------------------")
# print(vix)
# vix = yf.download("^INDIAVIX", start=df.index.min(), end=df.index.max())['Close'].shift(1).rename('VIX')

# Feature Engg

# Load and align VIX (lag vix)
vix = pd.read_csv('./VIX.csv', usecols=['Date', 'Price'], parse_dates=['Date'], index_col='Date', dayfirst=True)['Price'].rename('VIX').sort_index(ascending=True)
vix = vix.loc[vix.index <= df.index[-1]].shift(1)       #lag
vix_aligned = vix.reindex(df.index)


# residuals/volatility in expanding window

# Split returns into train/test first (time-based)
split_date = df.index[int(len(df) * 0.8)]
train_returns = df['Return'].iloc[1:].loc[:split_date]
test_returns = df['Return'].iloc[1:].loc[split_date:]

# storage for residuals/volatility (expanding window)
expanding_residuals = pd.Series(index=df.index, dtype=float)
expanding_cond_vol = pd.Series(index=df.index, dtype=float)

# GARCH.fit on training data
initial_model = arch_model(train_returns, vol="Garch", p=1, q=1, o=1, rescale=False)
initial_fit = initial_model.fit(disp='off')
expanding_residuals.loc[train_returns.index] = initial_fit.resid
expanding_cond_vol.loc[train_returns.index] = initial_fit.conditional_volatility

# refit GARCH iteratively (test set (expanding window))
for date in test_returns.index:
    current_data = df['Return'].iloc[1:].loc[:date]
    model = arch_model(current_data, vol="Garch", p=1, q=1, o=1, rescale=False)
    fit = model.fit(disp='off', update_freq=0)
    expanding_residuals.loc[date] = fit.resid[-1]                       # latest residual
    expanding_cond_vol.loc[date] = fit.conditional_volatility[-1]       # latest vol

# ----------------------------------------------------------------
# rebuild features with expanding window values

features = pd.DataFrame({
    'lag_resid': expanding_residuals.shift(1),  # expanding window
    'lag_vol': expanding_cond_vol.shift(1),     # expanding window
    'lag_sq_resid': (expanding_residuals.shift(1) ** 2),
    'negative_shock': (expanding_residuals.shift(1) < 0).astype(int),
    'VIX': vix_aligned  # lagged already
}).dropna()


# -----------------------------------------------------------------
# Recalculate target and prediction error

target = (df['Return'] ** 2).rename('realized_vol')
combined = features.join(target, how='inner').dropna()

combined['vol_pred_error'] = combined['realized_vol'] - expanding_cond_vol.reindex(combined.index)**2

# final features/target
features = combined.drop(columns=['realized_vol', 'vol_pred_error'])
target = combined['vol_pred_error']

# ------------------------------------------------------------
# date split

split_date = combined.index[int(len(combined) * 0.8)]
X_train = features.loc[features.index < split_date]
X_test = features.loc[features.index >= split_date]
y_train = target.loc[target.index < split_date]
y_test = target.loc[target.index >= split_date]


model = XGBRegressor(objective="reg:squarederror", n_estimators=200)
model.fit(X_train, y_train)


# Hybrid predictions (GARCH + XGBoost correction)
# garch_test_pred = garch_vol_pred.loc[y_test.index]            # gets changed to fit data leak issues
garch_test_pred = expanding_cond_vol.loc[y_test.index]**2

garch_test_pred_values = garch_test_pred.values

ml_correction = model.predict(X_test)
ml_correction = np.clip(
    ml_correction,
    a_min=-0.5 * garch_test_pred_values,
    a_max=0.5 * garch_test_pred_values
)


# hybrid_pred = garch_test_pred + ml_correction
hybrid_pred = pd.Series(
    garch_test_pred_values + ml_correction,
    index=garch_test_pred.index
)
hybrid_pred = hybrid_pred.clip(lower=1e-8)

# Mean Squared Error -- symmetric loss function
print(f"\nXGBoost MSE: {mean_squared_error(y_test, ml_correction):.4f}")
print(f"GARCH-Only MSE: {mean_squared_error(y_test, garch_test_pred):.4f}")
print(f"Hybrid MSE: {mean_squared_error(y_test, hybrid_pred):.4f}")

# QLIKE metric
def qlike_loss(actual, pred):                             # penalises under-prediction heavier than over-prediction (favours positive bias); asymmetric loss function
    epsilon = 1e-8
    pred_safe = np.clip(pred, a_min=epsilon, a_max=None)  # ensure pred > 0
    ratio = actual / pred_safe
    return np.mean(ratio - np.log(ratio) - 1)


print(f"\nGARCH-Only QLIKE: {qlike_loss(y_test, garch_test_pred):.4f}")
print(f"Hybrid QLIKE: {qlike_loss(y_test, hybrid_pred):.4f}")

# residual analysis
hybrid_residuals = y_test - hybrid_pred
# standardised_hybrid_residuals = hybrid_residuals / np.sqrt(hybrid_pred)
standardised_hybrid_residuals = hybrid_residuals / np.sqrt(np.clip(hybrid_pred, 1e-8, None))


test_stat, p_value = McLeodLi_test(standardised_hybrid_residuals, k=50)
print("\nMcLeod-Li Test (Hybrid):")
print(f"Test Stat: {test_stat:.2f}, p-value: {p_value:.4f}")

# validating predictions
print("Negative hybrid predictions:", (hybrid_pred < 0).sum())
print("Zero GARCH predictions:", (garch_test_pred <= 0).sum())

# plotting graphs
results = pd.DataFrame({
    "Observed": y_test,
    "GARCH-Only": garch_test_pred,
    "Hybrid": hybrid_pred
}, index=y_test.index)

plt.figure(figsize=(12, 6))
plt.plot(results.index, results["Observed"], label="Observed", alpha=0.6)
plt.plot(results.index, results["GARCH-Only"], label="GARCH-Only", linestyle='--')
plt.plot(results.index, results["Hybrid"], label="Hybrid", linestyle='-.')
plt.title("Volatility Forecast Comparison", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Volatility (σ²)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
