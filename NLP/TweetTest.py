import tweepy

tweets = []
tweetText = []

consumerKey = 'z9OQ8uZHDGce5AEsZahKSarMH'
consumerSecret = 'IvwzR3sLsg04wfFfYjPENWNHv5hYjeLiyrUl5SXn60ALz62zww'
accessToken = '1214740888678584320-C5qOOcZacPsnVjNygkI3x6VKzgCCbq'
accessTokenSecret = 'WoNHIvQpXv0IBvXx8H6KQOlWMmTaaEuMuB8HI9xnGfkjU'
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)


searchTerm = input("Enter Keyword/Tag to search about: ")
NoOfTerms = int(input("Enter how many tweets to search: "))


tweets = tweepy.Cursor(api.search, q=searchTerm,
                       lang="en").items(NoOfTerms)
for tweet in tweets:
     tweetText.append((tweet.text).encode('utf-8'))

print(tweetText)
