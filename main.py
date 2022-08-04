import pandas
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.linear_model import LinearRegression

data = pandas.read_csv('02.2 cost_revenue_dirty.csv')
data.describe()
x = DataFrame(data, columns =['production_budget_usd'])
y = DataFrame(data, columns =['worldwide_gross_usd'])

# plt.figure(figsize=(10,6))
# plt.scatter(x,y, alpha=0.3)
# plt.title('Film Cost vs Global Revenue')
# plt.xlabel('Production Budget $')
# plt.ylabel('Worldwide Gross $')
# plt.ylim(0,3000000000)
# plt.xlim(0, 450000000)
# plt.show()


regression = LinearRegression()
print(regression.fit(x,y))

print(regression.coef_)  #theta_1

#Intercept
regression.intercept_

plt.figure(figsize=(10,6))
plt.scatter(x,y, alpha=0.3)
plt.plot(x,regression.predict(x), color ='red', linewidth = 4)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0,3000000000)
plt.xlim(0, 450000000)
plt.show()

regression.score(x,y)