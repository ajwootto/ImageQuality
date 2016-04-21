from PIL import Image

def resize_image(name):
  im = Image.open(name)
  width, height = im.size   # Get dimensions
  new_width = new_height = min([width, height])
  left = (width - new_width)/2
  top = (height - new_height)/2
  right = (width + new_width)/2
  bottom = (height + new_height)/2

  im.crop((left, top, right, bottom))
  im.save(name)


for i in range(0, 978):
  resize_image('bingBad/' + str(i) + '.jpg')