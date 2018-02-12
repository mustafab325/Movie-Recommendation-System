import pandas as pd

movies = pd.read_csv("/Users/Documents/movies.csv")
ratings = pd.read_csv("/Users/Documents/ratings.csv")

movies = pd.merge(movies, ratings, on='movieId')

AvgRatings = pd.DataFrame(movies.groupby('movieId')['rating'].mean())
AvgRatings['vote_count'] = pd.DataFrame(movies.groupby('movieId')['rating'].count())

AvgRatings.rename(columns = {'rating':'average_rating'}, inplace = True)

AvgRatings.reset_index(inplace=True)

movies_orig = pd.read_csv('/Users/Documents/movies.csv')
AvgRatings = pd.merge(movies_orig, AvgRatings, on='movieId')

vote_counts = AvgRatings[AvgRatings['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = AvgRatings[AvgRatings['average_rating'].notnull()]['average_rating']

C = vote_averages.mean()
m = vote_counts.quantile(0.94)

SimpleRecommendation = AvgRatings[(AvgRatings['vote_count'] >= m) & (AvgRatings['vote_count'].notnull()) & (AvgRatings['average_rating'].notnull())][['movieId','title','genres','average_rating','vote_count']]

def weighted_rating(x):
    v = x['vote_count']
    R = x['average_rating']
    return (v/(v+m) * R) + (m/(m+v) * C)

SimpleRecommendation['wr'] = SimpleRecommendation.apply(weighted_rating, axis=1)
SimpleRecommendation.to_csv("output.csv", sep=',')
SimpleRecommendation = SimpleRecommendation.sort_values('wr', ascending=False)

print("Simple Recommendation")
print("The Top 100 Movies: ")
print(SimpleRecommendation['title'].head(15))






