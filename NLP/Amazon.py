import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf 
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
cf.go_offline()
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

pd.set_option('display.max_columns', None)
df=pd.read_csv("D:/Study Material/MAJOR_PROJECT/Web-Sentiment Analysis/analysis/NLP/amazon.csv")

# Removing coloumn name unnamed
df = df.sort_values("wilson_lower_bound",ascending=False)
df.drop('Unnamed: 0', inplace=True, axis=1)
df.head()

# Create function for missing values
def missing_values_analysis(df):
    na_columns_=[col for col in df.columns if df[col].isnull().sum()>0]
    n_miss=df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_=(df[na_columns_].isnull().sum()/df.shape[0]*100).sort_values(ascending=True)
    missing_df=pd.concat([n_miss,np.round(ratio_,2)],axis=1,keys=['Missing Values','Ratio'])
    missing_df=pd.DataFrame(missing_df)
    return missing_df

def check_dataframe(df,head=5,tail=5):
    print("SHAPE".center(82,'~'))
    print('Rows: {}'.format(df.shape[0]))
    print('columns: {}'.format(df.shape[1]))
    print("TYPES".center(82,'~'))
    print(df.dtypes)
    print("".center(82,'~'))
    print(missing_values_analysis(df))
    print('DUPLICATED VALUES'.center(83,'~'))
    print(df.duplicated().sum())
    print("QUANTITIES".center(82,'~'))
    print(df.quantile([0,0.05,0.50,0.95,0.99,1]).T)

# function to see unique values in each column
def check_class(dataframe):
    nunique_df=pd.DataFrame({'Variable':dataframe.columns,
                             'Classes':[dataframe[i].nunique()\
                                       for i in dataframe.columns]})
    nunique_df=nunique_df.sort_values('Classes',ascending=False)
    nunique_df=nunique_df.reset_index(drop=True)
    return nunique_df

# Catagorycal variable analysis for overall
constraints =['#B34D22','#EBE00C','#1FEB0C','#0C92EB','#DE5D83']
def categorical_variable_summary(df,column_name):
    fig=make_subplots(rows=1,cols=2,
                     subplot_titles=('Countplot','Percentage'),
                     specs=[[{'type': 'xy'},{'type': 'domain'}]])
    
    #making bar chart
    fig.add_trace(go.Bar(y=df[column_name].value_counts().values.tolist(),
                        x=[str(i) for i in df[column_name].value_counts().index],
                        text =df[column_name].value_counts().values.tolist(),
                        textfont=dict(size=14),
                        name=column_name,
                        textposition='auto',
                        showlegend=False,
                        marker=dict(color=constraints,
                                   line=dict(color='#DBE6DC',
                                            width=1))),
                  row=1, col=1)
    #making pie chart
    fig.add_trace(go.Pie(labels=df[column_name].value_counts().keys(),
                        values=df[column_name].value_counts().values,
                        textfont=dict(size=18),
                        textposition='auto',
                        showlegend=False,
                        name=column_name,
                        marker=dict(colors=constraints)),
                 row=1,col=2)
    fig.update_layout(title={'text': column_name,
                            'y':0.9,
                            'x':0.5,
                            'xanchor':'center',
                            'yanchor':'top'},
                     template='plotly_white')
    if(column_name=='overall'):
        fig.write_image("D:/Study Material/MAJOR_PROJECT/Web-Sentiment Analysis/analysis/NLP/image/graph0.png")
    else:
        fig.write_image("D:/Study Material/MAJOR_PROJECT/Web-Sentiment Analysis/analysis/NLP/image/graph.png")
        
  
# categorical_variable_summary(df,'overall')
df.reviewText.head()
review_example = df.reviewText[2031]

# saving text file
with open('D:/Study Material/MAJOR_PROJECT/Web-Sentiment Analysis/analysis/NLP/sample.txt', 'w') as f:
    f.write(review_example)
    
def NLP():
    text = open('D:/Study Material/MAJOR_PROJECT/Web-Sentiment Analysis/analysis/NLP/sample.txt', encoding='utf-8').read()
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

    # Using word_tokenize because it's faster than split()
    tokenized_words = word_tokenize(cleaned_text, "english")

    # Removing Stop Words
    final_words = []
    for word in tokenized_words:
        if word not in stopwords.words('english'):
            final_words.append(word)

    # Lemmatization - From plural to single + Base form of a word (example better-> good)
    lemma_words = []
    for word in final_words:
        word = WordNetLemmatizer().lemmatize(word)
        lemma_words.append(word)

    emotion_list = []
    with open('D:/Study Material/MAJOR_PROJECT/Web-Sentiment Analysis/analysis/NLP/emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')

            if word in lemma_words:
                emotion_list.append(emotion)


    w = Counter(emotion_list)
    print(w)



    def sentiment_analyse(sentiment_text):
        score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
        if score['neg'] > score['pos']:
            print("Negative Sentiment")
        elif score['neg'] < score['pos']:
            print("Positive Sentiment")
        else:
            print("Neutral Sentiment")


    sentiment_analyse(cleaned_text)

    fig, ax1 = plt.subplots()
    ax1.bar(w.keys(), w.values())
    fig.autofmt_xdate()
    plt.savefig('D:/Study Material/MAJOR_PROJECT/Web-Sentiment Analysis/analysis/NLP/image/graph1.png')
    plt.show()

review_example = re.sub("[^a-zA-Z]",'',review_example)
review_example = review_example.lower().split()

#make all rows same in lower case
rt = lambda x: re.sub("[^a-zA-Z]",' ',str(x))
df["reviewText"]=df["reviewText"].map(rt)
df["reviewText"]=df["reviewText"].str.lower()
# df.head()

# ALL THE DATA IS SORTED NOW TIME TO DO SENTIMENT ANALYSIS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
df[['polarity','subjectivity']]=df['reviewText'].apply(lambda Text:pd.Series(TextBlob(Text).sentiment))

for index, row in df['reviewText'].iteritems():
    score=SentimentIntensityAnalyzer().polarity_scores(row)
    neg=score['neg']
    neu=score['neu']
    pos=score['pos']
    if neg>pos:
        df.loc[index,'sentiment']="Negative"
    elif pos>neg:
        df.loc[index,'sentiment']="Positive"
    else:
        df.loc[index,'sentiment']="Netural"

df[df['sentiment']=='Positive'].sort_values("wilson_lower_bound",
                                           ascending=False).head(5)

categorical_variable_summary(df,'sentiment')
categorical_variable_summary(df,'overall')
NLP()

