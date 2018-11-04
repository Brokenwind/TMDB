import os
import numpy as np
import pandas as pd
import json
from wordcloud import wordcloud
import matplotlib.pyplot as plt


def loadData():
    # 加载数据
    credits = pd.read_csv('data/tmdb_5000_credits.csv')
    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    # credits.head()
    # movies.head()
    # credits 和 movies 中都有title 字段，删除其中一个
    del credits['title']
    # 做链接方式合并数据
    total = pd.merge(movies, credits, left_on='id', right_on='movie_id', how='left')
    return total


def getName(x):
    name = []
    for item in x:
        name.append(item['name'])
    return '|'.join(name)


def cleanData(total):
    # 不需要字段
    drop_title = ['homepage', 'id', 'original_language', 'original_title', 'overview', 'spoken_languages', 'status',
                  'tagline', 'movie_id', 'cast', 'crew']
    total.drop(drop_title, axis=1, inplace=True)
    # release_date列有1条缺失数据，将其查找出来并填充：
    total.release_date = total.release_date.fillna('2014-06-01')
    # runtime列有2条缺失数据，将其查找出来并填充：
    total.runtime = total.runtime.fillna(94, limit=1)
    total.runtime = total.runtime.fillna(240, limit=1)

    total.release_date = pd.to_datetime(total.release_date, format='%Y-%m-%d', errors='coerce').dt.year

    json_column = ['genres', 'keywords', 'production_companies', 'production_countries']
    for column in json_column:
        total[column] = total[column].map(json.loads)

    for column in json_column:
        total[column] = total[column].map(getName)

    print(total.info())


def get_genre_set(total):
    genre_set = set()
    for item in total['genres']:
        genre_set.update(item.split('|'))
    genre_set.discard('')
    print("genres num: %d" % len(genre_set))
    return genre_set


def get_data_genre(total):
    genre_set = get_genre_set(total)
    genre_df = pd.DataFrame()
    for genre in genre_set:
        genre_df[genre] = total['genres'].str.contains(genre).map(lambda x: 1 if x else 0)

    genre_df['release_date'] = total['release_date']

    return genre_df


def getGenreYear(total):
    '''
    获取种类和时间之间的关系
    :param total:
    :return:
    '''
    genre_df = get_data_genre(total)
    genre_by_year = genre_df.groupby('release_date').sum()

    return genre_by_year


def getAndShowGenreYear(total):
    '''
    获取类别和时间之间的关系
    :param total:
    :return:
    '''
    genre_by_year = getGenreYear(total)
    showGenreYeas(genre_by_year)


def showGenreYeas(genre_by_year):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure(figsize=(12, 6))
    plt.plot(genre_by_year, label=genre_by_year.columns)
    plt.legend(genre_by_year)
    plt.title("电影类型随时间变化趋势", fontsize=20)
    plt.xticks(range(1910, 2018, 10))
    plt.xlabel("年份", fontsize=20)
    plt.ylabel("数量", fontsize=20)
    plt.show()


def getGenreCount(total):
    '''获取类别统计信息
    '''
    genre_by_year = getGenreYear(total)
    genre_sum = genre_by_year.sum(axis=0).sort_values(ascending=True)
    return genre_sum


def getAndShowGenreCount(total):
    '''获取类别统计信息并展示
    '''
    genre_sum = getGenreCount(total)
    showGenSum(genre_sum)


def showGenSum(genre_sum):
    print(genre_sum)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    genre_sum.plot.barh(label='genre', figsize=(12, 6))
    plt.title("电影类型分布图", fontsize=20)
    plt.xlabel("数量", fontsize=20)
    plt.ylabel("电影类型", fontsize=20)
    plt.show()


def getGenreProfit(total):
    profit_df = pd.DataFrame()
    total['profit'] = total['revenue'] - total['budget']
    genre_set = get_genre_set(total)
    genre_df = get_data_genre(total)
    profit_df = pd.concat([genre_df.iloc[:, :-1], total['profit']], axis=1)
    profit_by_genre = pd.Series(index=genre_set)
    for genre in genre_set:
        # 每次循环按照一个类型进行划分，分成是这一类型和不是这一类型两种，然后分别计算这两类的平均值，然后取是这一类的均值
        # 注意 loc[1],指的是索引为1的那一行，而不是行号
        profit_by_genre[genre] = profit_df.groupby(genre)['profit'].mean().loc[1]
    profit_by_genre = profit_by_genre.sort_values(ascending=False)

    print(profit_by_genre)

    return profit_by_genre


def showGenreProfit(profit_by_genre):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    profit_by_genre.plot.barh(label='genre', figsize=(12, 6))
    plt.title('电影类型利润分布图', fontsize=20)
    plt.xlabel('利润', fontsize=20)
    plt.ylabel('电影类型', fontsize=20)
    plt.show()


def getAndShowGenreProfit(total):
    profit_by_genre = getGenreProfit(total)
    showGenreProfit(profit_by_genre)


def get_original_recompose(total):
    original_recompose = pd.DataFrame()
    original_recompose['type'] = total['keywords'].str.contains('based on novel').map(lambda x: 1 if x else 0)
    count_list = original_recompose.groupby('type')['type'].count()
    original_recompose['profit'] = total['profit']
    original_recompose['budget'] = total['budget']
    print(original_recompose.info())
    original_recompose = original_recompose.groupby('type').mean()
    original_recompose['count'] = count_list
    original_recompose['profit_rate'] = original_recompose['profit'] / original_recompose['budget']
    original_recompose.rename(index={0: 'original', 1: 'recompose'}, inplace=True)
    print(original_recompose)
    return original_recompose


def showOriRecom1(original_recomposee):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure(figsize=(12, 6))
    data_count = original_recomposee.loc[original_recomposee.index, 'count']
    plt.bar(original_recomposee.index, data_count)
    plt.xlabel("原创-改编")
    plt.ylabel("数量")
    plt.show()


def showOriRecom2(original_recomposee):
    # 绘制利润柱状图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    x = list(range(len(original_recomposee.index)))
    index = original_recomposee.index
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.bar(index, original_recomposee['profit'])
    ax1.set_title('原创与改编的利润对比')
    ax1.set_ylabel('利润')
    ax1.set_xlabel('原创-改编')
    # 绘制利润率折线图
    import matplotlib.ticker as mtick
    ax2 = ax1.twinx()
    ax2.plot(index, original_recomposee['profit_rate'], 'ro-', lw=2)
    fmt = '%.2f%%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax2.yaxis.set_major_formatter(yticks)
    ax2.set_ylabel('利润率')
    plt.show()


def getAndShowOriRcom(total):
    original_recomposee = get_original_recompose(total)
    # showOriRecom(original_recomposee)
    showOriRecom2(original_recomposee)


def getCountryCount(total):
    country_set = set()
    for x in total['production_countries']:
        country_set.update(x.split('|'))
    country_set.discard('')
    country_df = pd.DataFrame()
    for country in country_set:
        country_df[country] = total['production_countries'].str.contains(country).map(lambda x: 1 if x else 0)
    # 数据类型是Series
    country_df = country_df.sum().sort_values(ascending=False)
    return country_df


def showCountryCount(country_df):
    rate = country_df / country_df.sum()
    others = 0.01
    # 只留下比例大于0.01的国家
    rate1 = rate[rate >= others]
    # 占比小于0.01的 都归于 其它
    rate1['others'] = rate[rate < others].sum()
    # 占比大于4%的向外延伸
    explode = (rate1 >= 0.04) / 20 + 0.02
    rate1.plot.pie(autopct='%1.1f%%', figsize=(10, 10), explode=explode, label='')
    plt.title('电影产地分布图')
    plt.show()


def getAndShowCountryCount(total):
    country_df = getCountryCount(total)
    showCountryCount(country_df)


def get_and_show_keywords(total):
    keywords_list = []
    for item in total['keywords']:
        keywords_list.append(item)
    keywords_list = '|'.join(keywords_list)
    wc = wordcloud.WordCloud(background_color='black', max_words=3000, scale=1.5).generate(keywords_list)
    plt.figure(figsize=(14, 8))
    plt.imshow(wc)
    plt.axis('off')
    plt.show()


def get_popular_genre(total):
    popular_genre_df = pd.DataFrame()
    genre_set = get_genre_set(total)
    genre_df = get_data_genre(total)

    popular_genre_df = pd.concat([genre_df.iloc[:, :-1], total['popularity']], axis=1)
    popular_list = []
    for genre in genre_set:
        popular_list.append(popular_genre_df.groupby(genre)['popularity'].mean().loc[1])
    popular_genre = pd.DataFrame(index=genre_set)
    popular_genre['popularity'] = popular_list
    print(popular_genre)
    return popular_genre


def show_popular_genre(popular_genre):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    popular_genre.sort_values(by='popularity').plot.barh(label='genre')
    plt.title('电影类型受欢迎分布')
    plt.xlabel('受欢迎程度')
    plt.ylabel('电影类型')
    plt.show()


def get_and_show_popular_genre(total):
    popular_genre = get_popular_genre(total)
    show_popular_genre(popular_genre)


def get_and_show_popular_runtime(total):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.scatter(total['runtime'], total['popularity'])
    plt.title('电影时长和欢迎度的分布')
    plt.xlabel('电影时长')
    plt.ylabel('受欢迎程度')
    plt.show()


def get_and_show_popular_vote(total):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.scatter(total['vote_average'], total['popularity'])
    plt.title('电影评分和欢迎度的分布')
    plt.xlabel('电影评分')
    plt.ylabel('受欢迎程度')
    plt.show()


def get_company_data(total):
    company_list = ['Universal Pictures', 'Paramount Pictures']
    genre_df = get_data_genre(total)
    company_df = pd.DataFrame()
    for company in company_list:
        company_df[company] = total['production_companies'].str.contains(company).map(lambda x: 1 if x else 0)
    company_df = pd.concat([company_df, genre_df.iloc[:, :-1], total['profit']], axis=1)
    return company_list, company_df


def get_company_genre(total):
    genre_set = get_genre_set(total)
    company_list, company_df = get_company_data(total)
    # 当设置了索引是哪些的时候，设置数据时就会对应填入
    company_genre = pd.DataFrame(index=genre_set)
    for company in company_list:
        company_genre[company] = company_df.groupby(company).sum().loc[1]
        company_genre[company]
    return company_genre

def show_company_genre(company_genre):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(1,2,1)
    plt.pie(company_genre['Universal Pictures'],labels=company_genre.index,autopct='%.1f%%')
    plt.title('Universal Pictures')

    ax2 = fig.add_subplot(1,2,2)
    plt.pie(company_genre['Paramount Pictures'],labels=company_genre.index,autopct='%.1f%%')
    plt.title('Paramount Pictures')

    plt.show()

def get_and_show_company_genre(total):
    company_genre = get_company_genre(total)
    show_company_genre(company_genre)

if __name__ == '__main__':
    total = loadData()
    cleanData(total)
    # getAndShowGenreYear(total)
    getGenreProfit(total)
    # getAndShowGenreProfit(total)
    # get_original_recompose(total)
    # getAndShowOriRcom(total)
    # getCountryCount(total)
    # getAndShowCountryCount(total)
    # get_and_show_keywords(total)
    # get_popular_genre(total)
    # get_and_show_popular_genre(total)
    # get_and_show_popular_runtime(total)
    #get_and_show_popular_vote(total)
    #get_company_genre(total)
    get_and_show_company_genre(total)
