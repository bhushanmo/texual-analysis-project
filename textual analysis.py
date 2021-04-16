from collections import Counter
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm

# initialise empty lists
ticker = []
positive_count = []
negative_count = []


# function to read positive and negative word lists
def mcdonald(filename):
    fle = open(filename)
    words = [line.rstrip() for line in fle.readlines()]
    return words


# apply the function on word lists
positive_words = mcdonald(r"./data/positive_german_words.csv")
negative_words = mcdonald(r"./data/negative_german_words.csv")


# function to count positive and negative words in text reports
def process(fi):
    with open(fi, 'r', encoding="utf-8") as file:

        data = file.read()

        count = Counter(data.split())

        positive = 0
        negative = 0
        for i, value in count.items():
            i = i.rstrip('.,?!\n')
            if i in positive_words:
                positive += value
            if i in negative_words:
                negative += value

    # extract ticker from text filenames
    fullname = os.path.basename(file.name)
    name_token = fullname.split("_")
    sym = name_token[1]

    # append word counts and ticker to lists
    positive_count.append(positive)
    negative_count.append(negative)
    ticker.append(sym)


# loop process over entire directory
path = r".\data\annual_reports\by_year\2015"
for f in os.listdir(path):
    v = (os.path.join(path, f))
    process(v)

# join word count and ticker lists to form dataframe
df1 = pd.DataFrame(list(zip(ticker, positive_count, negative_count)),
                   columns=['ticker', 'positive_count', 'negative_count'])

# get mean of word counts
wrd_ct = df1.groupby(['ticker'])[['positive_count', 'negative_count']].mean().reset_index()
print(wrd_ct)

# preparing stock data
stock_data = pd.read_csv(r'.\data\stock_price_data\german_monthly_stock_data.csv', low_memory=False)
pd.options.mode.chained_assignment = None
stock_data['yearmonthday'] = stock_data['yearmonthday'].astype(str)
data2015 = stock_data[stock_data['yearmonthday'].str.contains(pat='2015')]  # selecting data for year 2015
data2 = data2015[data2015['ret_usd'].notna()]

# mean of stock return for 2015
average2015 = data2.groupby(['Name', 'isin', 'bloomberg_ticker'])['ret_usd'].mean().reset_index()
average2015 = average2015[average2015.bloomberg_ticker != '#N/A Invalid Security']  # drop 'N/A invalid security' names
tickerstrip = average2015.bloomberg_ticker.str.strip(" GY Equity")  # strip off GY equity from bloomberg ticker
average2015['ticker'] = tickerstrip
average2015['ticker'] = average2015['ticker'].astype(str)
print(average2015)

# join wrd_ct and stock data over common ticker symbol
final = pd.merge(average2015, wrd_ct, how='inner', on='ticker')
# calculate 'sentiment_count' column as (positive - negative) word count
final["sentiment_count"] = final["positive_count"] - final["negative_count"]
# calculate 'sentiment_ratio' column as (positive / negative) word count
final["sentiment_ratio"] = final["positive_count"] / final["negative_count"]
print(final)


# Descriptive statistics on stock return, sentiment_count, sentiment_ratio
print(final['ret_usd'].describe())
print(final['sentiment_count'].describe())
print(final['sentiment_ratio'].describe())
sns.boxplot(x='ret_usd', data = final, orient='v')
plt.title('stock return boxplot')
plt.show()
sns.boxplot(x='sentiment_count', data = final, orient='v')
plt.title('sentiment count boxplot')
plt.show()
sns.boxplot(x='sentiment_ratio', data = final, orient='v')
plt.title('sentiment ratio boxplot')
plt.show()
sns.distplot(final['ret_usd'], kde= False)
plt.title('stock return histogram')
plt.show()
sns.distplot(final['sentiment_count'], kde= False)
plt.title('sentiment count histogram')
plt.show()
sns.distplot(final['sentiment_ratio'], kde= False)
plt.title('sentiment ratio histogram')
plt.show()


# Results for stock return vs 'sentiment_count'
sns.lmplot(x='sentiment_count', y='ret_usd', data=final, fit_reg=True)  # scatter plot
plt.title('stock return vs sentiment_count')
plt.show()
corr = (final['ret_usd'].corr(final['sentiment_count']))  # correlation value between return and sentiment_count
print("correlation coefficient:", corr)
result = sm.ols(formula="ret_usd ~ sentiment_count", data=final).fit()  # OLS regression
print(result.params)
print(result.summary())


# Results for stock return vs 'sentiment_ratio'
sns.lmplot(x='sentiment_ratio', y='ret_usd', data=final, fit_reg=True)  # scatter plot
plt.title('stock return vs sentiment_ratio')
plt.show()
corr = (final['ret_usd'].corr(final['sentiment_ratio']))  # correlation value between return and sentiment_ratio
print("correlation coefficient:", corr)
result = sm.ols(formula="ret_usd ~ sentiment_ratio", data=final).fit()  # OLS regression
print(result.params)
print(result.summary())

