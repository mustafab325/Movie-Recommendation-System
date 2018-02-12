try:
    import pandas as pd
    import numpy as np
    import sys
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
    from ast import literal_eval
    from nltk.stem.snowball import SnowballStemmer
    import warnings; warnings.simplefilter('ignore')
    import random
except:
    ImportError

reload(sys)
sys.setdefaultencoding('utf8')

# Reading data sets
try:
    links_small = pd.read_csv('/Users/Documents/links_small.csv')
    md = pd.read_csv('/Users/Documents/movies_metadata.csv')
    credit = pd.read_csv('/Users/Documents/credits.csv')
    keywords = pd.read_csv('/Users/Documents/keywords.csv')
except:
    print("Failed to read one or more data sets. Please check file path and try again")


# Reading total votes and average votes from movies_metadata file
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')


C = vote_averages.mean()
m = vote_counts.quantile(0.95)

# Reading genres and year of the movie
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][
    ['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')

# The IMDB weighted rating formula
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)
s = md.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genres'
gen_md = md.drop('genres', axis=1).join(s)

links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
md = md.drop([19730, 29503, 35587])

##Merging credit and keywords with link_smalls
md['id'] = md['id'].astype('int')
md = md.merge(credit, on='id')
md = md.merge(keywords, on='id')
data = md[md['id'].isin(links_small)].copy()

##Creating a copy of dataframe to avoid settingwithcopywarning
##sub_smd = smd[['tagline', 'overview','keywords','cast','crew']].copy()

##Performing missing value treatment
data['tagline'] = data['tagline'].fillna('')
##Creating the columns description by concatenating overview and tagline columns
data['description'] = data['overview'] + data['tagline']

####Performing missing value treatment if description column
data['description'] = data['description'].fillna('')

####Applied literal_eval on cast, crew, keywords
data['cast'] = data['cast'].apply(literal_eval)
data['crew'] = data['crew'].apply(literal_eval)
data['keywords'] = data['keywords'].apply(literal_eval)

##Lambda function to obtain len of above lists
data['cast_size'] = data['cast'].apply(lambda x: len(x))
data['crew_size'] = data['crew'].apply(lambda x: len(x))


def get_director(x):
    for i in x:

        if i['job'] == 'Director':
            return i['name']
    return np.nan

data['director'] = data['crew'].apply(get_director)
##Lambda function to get names of cast members and to obtain top 3 member names
data['cast'] = data['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
data['cast'] = data['cast'].apply(lambda x: x[:5] if len(x) >= 5 else x)
data['keywords'] = data['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
data['cast'] = data['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])



# Increase weight of director so movies recommended will be heavily based on the director
data['director'] = data['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
data['director'] = data['director'].apply(lambda x: [x,x, x])



keywords['id'] = keywords['id'].astype('int')
credit['id'] = credit['id'].astype('int')
data['id'] = data['id'].astype('int')

s = data.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s = s[s > 1]
stemmer = SnowballStemmer('english')


def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)

    return words

data['keywords'] = data['keywords'].apply(filter_keywords)


data['keywords'] = data['keywords'].apply(lambda x: [stemmer.stem(i.decode('cp850').replace(u"\u2019", u"\x27")) for i in x])
data['keywords'] = data['keywords'].apply(lambda x: [unicode.lower(i.replace(" ", "")) for i in x])
data['soup'] = data['keywords']+ data['cast'] + data['director']+data['genres']

data['soup'] = data['soup'].apply(lambda x: ' '.join(x))

count = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(data['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

data = data.reset_index()
titles = data['title']
indices = pd.Series(data.index, index=data['title'])


def improved_recommendations(title):
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]

        movies = data.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
        vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(0.60)
        qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')
        qualified['wr'] = qualified.apply(weighted_rating, axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(10)
        return qualified['title']
    except:
        print "Movie does not exist in the database. Please try again with a different movie"

def main():
    flag = 0
    while flag == 0:
        user_name = raw_input("\nPlease enter your name (Q! to quit):")
        if user_name == "Q!":
            flag = 1
            break
        else:
    #         print improved_recommendations(title_input)
            read_from_file(user_name)


def read_from_file(user_name):
    file_name = user_name + ".txt"

    file_obj = open(file_name, 'rb')

    movies = []
    for movie in file_obj:
        movies.append(movie.replace("\r\n", "").replace("\r", ""))

    file_obj.close()

    # print movies

    final_list = []
    for movie in movies:
        movie_list = improved_recommendations(movie)
        for item in movie_list:
            final_list.append(item)
    # x = random.shuffle(final_list)
    final_suggestions = sorted(final_list, key=lambda k: random.random())
    for suggestions in range(0,10):
        print final_suggestions[suggestions]
    # print final_list
main()
