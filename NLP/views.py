from django.shortcuts import render
from NLP.models import Contact
import string
from collections import Counter
from datetime import datetime
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys, re
from textblob import TextBlob
import tweepy

# Amazon libraries
import numpy as np
import pandas as pd
import nltk
from wordcloud import WordCloud
import seaborn as sns
import cufflinks as cf 
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
cf.go_offline()
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# Creating your views here.
def index(request):
    return render(request, 'index.html')

def started(request):
    return render(request, 'started.html')

def text(request):
    return render(request, 'text.html')

def twitter(request):
    return render(request, 'twitter.html')

def contact(request):
    if request.method == 'POST':
        email = request.POST.get("email")
        name = request.POST.get("name")
        query = request.POST.get("queries")
        contact = Contact(name=name, email=email, query=query, date = datetime.today())
        contact.save()
    return render(request, 'contact.html')

def amazon(request):
    return render(request, 'amazon.html')

def result(request):
    if request.method == 'GET':
        text = request.GET.get("queries")
        lower_case = text.lower()
        cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

        
        tokenized_words = word_tokenize(cleaned_text, "english")

        
        final_words = []
        for word in tokenized_words:
            if word not in stopwords.words("english"):
                final_words.append(word)
        
        
        lemma_words = []
        for word in final_words:
            word = WordNetLemmatizer().lemmatize(word)
            lemma_words.append(word)
        
        emotion_list = []
        with open('emotions.txt', 'r') as file:
            for line in file:
                clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
                word, emotion = clear_line.split(':')
        
                if word in lemma_words:
                    emotion_list.append(emotion)
        
        
        w = Counter(emotion_list)        
        
        
        
        fig, ax1 = plt.subplots()
        ax1.bar(w.keys(), w.values())
        fig.autofmt_xdate()
        plt.title('Feelings From the text you entered')
        plt.savefig('D:/Study Material/MAJOR_PROJECT/Web-Sentiment Analysis/analysis/statics/Text1Analysis.png')
        plt.close()
    return render(request, 'result.html')

def result2(request):
    if request.method == 'GET':  
        def DownloadData():
           
            consumerKey = 'fW7J2DpGStPAiTQiBxt9EoYOY'
            consumerSecret = 'FHqs8nQGH48tZqoSZBMI3nNyCGIgmXbFydmE67c5sv0nlpB4oL'
            accessToken = '1214740888678584320-Mlhpg6n0HhAOz2dlhOUvdQLvNXF9BH'
            accessTokenSecret = 'k43gRrVrNormBPFQwNY3cLIMYbdV10xJJ2aAdVcZf8tOI'
            auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
            auth.set_access_token(accessToken, accessTokenSecret)
            api = tweepy.API(auth)
        
            
            searchTerm = request.GET.get("hashtag")
            NoOfTerms = int(request.GET.get("tweetnumber"))
        
            
            tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en").items(NoOfTerms)
            for tweet in tweets:
                tweetText.append((tweet.text).encode('utf-8'))
            return(tweetText)
    
        def TweetAnalysis():
            
            lower_case = text.lower()
        
           
            cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

        
            tokenized_words = word_tokenize(cleaned_text, "english")

        
            final_words = []
            for word in tokenized_words:
                if word not in stopwords.words("english"):
                    final_words.append(word)
        
        
            lemma_words = []
            for word in final_words:
                word = WordNetLemmatizer().lemmatize(word)
                lemma_words.append(word)
    
            
            emotion_list = []
            with open('emotions.txt', 'r') as file:
                for line in file:
                    clear_line = line.replace('\n', '').replace(',', '').replace("'", '').strip()
                    word, emotion = clear_line.split(':')
                    if word in lemma_words:
                        emotion_list.append(emotion)
        
        
            w = Counter(emotion_list)
        
            fig, ax1 = plt.subplots()
            ax1.bar(w.keys(), w.values())
            fig.autofmt_xdate()
            plt.title('How people are Feeling..')
            plt.savefig('D:/Study Material/MAJOR_PROJECT/Web-Sentiment Analysis/analysis/statics/graph.png')
            plt.close()

        tweets = []
        tweetText = []
        DownloadData()
        text_1 = b" ".join(tweetText)
        text = text_1.decode()
        TweetAnalysis()

        
        def Tweetpie():
            class SentimentAnalysis:
                def DownloadTweet(self):
                    self.tweets = []
                    self.tweetText = []

                    
                    consumerKey = 'fW7J2DpGStPAiTQiBxt9EoYOY'
                    consumerSecret = 'FHqs8nQGH48tZqoSZBMI3nNyCGIgmXbFydmE67c5sv0nlpB4oL'
                    accessToken = '1214740888678584320-Mlhpg6n0HhAOz2dlhOUvdQLvNXF9BH'
                    accessTokenSecret = 'k43gRrVrNormBPFQwNY3cLIMYbdV10xJJ2aAdVcZf8tOI'
                    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
                    auth.set_access_token(accessToken, accessTokenSecret)
                    api = tweepy.API(auth)

                   
                    searchTerm = request.GET.get("hashtag")
                    NoOfTerms = int(request.GET.get("tweetnumber"))

                    
                    self.tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en").items(NoOfTerms)



                    
                    polarity = 0
                    positive = 0
                    wpositive = 0
                    spositive = 0
                    negative = 0
                    wnegative = 0
                    snegative = 0
                    neutral = 0


                   
                    for tweet in self.tweets:
                       
                        self.tweetText.append(self.cleanTweet(tweet.text).encode('utf-8'))
                     
                        analysis = TextBlob(tweet.text)
                      
                        polarity += analysis.sentiment.polarity  # adding up polarities to find the average later

                        if (analysis.sentiment.polarity == 0):  # adding reaction of how people are reacting to find average later
                            neutral += 1
                        elif (analysis.sentiment.polarity > 0 and analysis.sentiment.polarity <= 0.3):
                            wpositive += 1
                        elif (analysis.sentiment.polarity > 0.3 and analysis.sentiment.polarity <= 0.6):
                            positive += 1
                        elif (analysis.sentiment.polarity > 0.6 and analysis.sentiment.polarity <= 1):
                            spositive += 1
                        elif (analysis.sentiment.polarity > -0.3 and analysis.sentiment.polarity <= 0):
                            wnegative += 1
                        elif (analysis.sentiment.polarity > -0.6 and analysis.sentiment.polarity <= -0.3):
                            negative += 1
                        elif (analysis.sentiment.polarity > -1 and analysis.sentiment.polarity <= -0.6):
                            snegative += 1



                    # finding average of how people are reacting
                    positive = self.percentage(positive, NoOfTerms)
                    wpositive = self.percentage(wpositive, NoOfTerms)
                    spositive = self.percentage(spositive, NoOfTerms)
                    negative = self.percentage(negative, NoOfTerms)
                    wnegative = self.percentage(wnegative, NoOfTerms)
                    snegative = self.percentage(snegative, NoOfTerms)
                    neutral = self.percentage(neutral, NoOfTerms)

                    # finding average reaction
                    polarity = polarity / NoOfTerms

                    self.plotPieChart(positive, wpositive, spositive, negative, wnegative, snegative, neutral, searchTerm, NoOfTerms)


                def cleanTweet(self, tweet):
                    # Remove Links, Special Characters etc from tweet
                    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())

                
                def percentage(self, part, whole):
                    temp = 100 * float(part) / float(whole)
                    return format(temp, '.2f')

                def plotPieChart(self, positive, wpositive, spositive, negative, wnegative, snegative, neutral, searchTerm, noOfSearchTerms):
                    labels = ['Positive [' + str(positive) + '%]', 'Weakly Positive [' + str(wpositive) + '%]','Strongly Positive [' + str(spositive) + '%]', 'Neutral [' + str(neutral) + '%]',
                              'Negative [' + str(negative) + '%]', 'Weakly Negative [' + str(wnegative) + '%]', 'Strongly Negative [' + str(snegative) + '%]']
                    sizes = [positive, wpositive, spositive, neutral, negative, wnegative, snegative]
                    colors = ['yellowgreen','lightgreen','darkgreen', 'gold', 'red','lightsalmon','darkred']
                    explode = [0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07]
                    patches, texts = plt.pie(sizes, colors=colors, explode=explode, shadow=True, wedgeprops={'edgecolor':'black'})
                    plt.legend(patches, labels, loc="best")
                    plt.title('How people are reacting on ' + searchTerm + ' by analyzing ' + str(noOfSearchTerms) + ' Tweets.')
                    plt.axis('equal')
                    plt.tight_layout()
                    plt.savefig('D:/Study Material/MAJOR_PROJECT/Web-Sentiment Analysis/analysis/statics/Tweet2Pie.png')
                    plt.close()

            sa = SentimentAnalysis()
            sa.DownloadTweet()
        Tweetpie()
    return render(request, 'result2.html')


def result3(request):
    if request.method=='GET':
        User_id = int(request.GET.get("User_id"))

        pd.set_option('display.max_columns', None)
        df=pd.read_csv("D:/Study Material/MAJOR_PROJECT/Web-Sentiment Analysis/analysis/NLP/amazon.csv")

        # Removing coloumn name unnamed
        df = df.sort_values("wilson_lower_bound",ascending=False)
        df.drop('Unnamed: 0', inplace=True, axis=1)
        # df.head()

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
                fig.write_image("D:/Study Material/MAJOR_PROJECT/Web-Sentiment Analysis/analysis/statics/images/graph0.png")
            else:
                fig.write_image("D:/Study Material/MAJOR_PROJECT/Web-Sentiment Analysis/analysis/statics/images/graph.png")
        
  
        # categorical_variable_summary(df,'overall')
        df.reviewText.head()
        review_example = df.reviewText[User_id]

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
            plt.savefig('D:/Study Material/MAJOR_PROJECT/Web-Sentiment Analysis/analysis/statics/images/graph1.png')
            plt.show()

        review_example = re.sub("[^a-zA-Z]",'',review_example)
        review_example = review_example.lower().split()

        #make all rows same in lower case
        rt = lambda x: re.sub("[^a-zA-Z]",' ',str(x))
        df["reviewText"]=df["reviewText"].map(rt)
        df["reviewText"]=df["reviewText"].str.lower()

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
    return render(request, 'result3.html')
    