import praw 
import urllib
import os

r = praw.Reddit(user_agent='OSX:Good_Images_Scraper:1.0 (by /u/superadamwo)')
submissions = r.get_subreddit('itookapicture').get_top_from_month(limit=200)

photo_directory = 'redditGood'

if not os.path.exists(photo_directory):
    os.makedirs(photo_directory)

index = 404
for x in submissions:
    if 'imgur.com' in x.url:
      if '.jpg' not in x.url:
        x.url += ".jpg"
    else:
      continue

    print index
    print x.url
    
    score = str(int(float(x.ups/2300.0) * 100))

    urllib.urlretrieve(x.url, photo_directory + '/' + str(index) + '_' + score + '.jpg', )
    index += 1

