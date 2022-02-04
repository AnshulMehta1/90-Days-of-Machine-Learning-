from PIL import Image
import glob
import os

lst_imgs = [i for i in glob.glob("*.jpg")]
 
# It creates a folder called ltl if does't exist
if not "resize" in os.listdir():
	os.mkdir("resize")
 
print(lst_imgs)
for i in lst_imgs:
	img = Image.open(i)
	img = img.resize((300, 300), Image.ANTIALIAS)
	img.save("resize\\" + i[:-4] + "_resized.png")
 
 
print("Done")
os.startfile("resize")
