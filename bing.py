from py_bing_search import PyBingSearch
import urllib
import urllib2
import json
import os
import socket

socket.setdefaulttimeout(5)

key = '4axpjG94pE8x9yUZqveY2LObcgNLVfX5oTW6+s5JbR0'
bing = PyBingSearch('4axpjG94pE8x9yUZqveY2LObcgNLVfX5oTW6+s5JbR0')

credentialBing = 'Basic ' + (':%s' % key).encode('base64')[:-1] # the "-1" is to remove the trailing "\n" which encode adds

photo_directory = 'bingBad'
if not os.path.exists(photo_directory):
    os.makedirs(photo_directory)

for offset in range(1050, 50000, 50):
  bing_search_url = "https://api.datamarket.azure.com/Bing/Search/v1/Image?Query=%27bad%20photography%27&$format=json&$top=200&$skip=" + str(offset)

  request = urllib2.Request(bing_search_url)
  request.add_header('Authorization', credentialBing)
  requestOpener = urllib2.build_opener()
  response = requestOpener.open(request) 

  results = json.load(response)

  for i, image in enumerate(results['d']['results']):
    print i + offset
    img_url = image['Thumbnail']['MediaUrl']
    urllib.urlretrieve(img_url, photo_directory + '/' + str(i + offset) + '.jpg', )
