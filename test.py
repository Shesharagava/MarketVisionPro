import snscrape.modules.twitter as sntwitter

for i, tweet in enumerate(sntwitter.TwitterSearchScraper("Tesla").get_items()):
    print(tweet.content)
    if i > 3:
        break
