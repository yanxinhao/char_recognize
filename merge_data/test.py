from PIL import Image
import pytesseract
import os

def initTable(threshold=140):
	table=[]
	for i in range(256):
		if i < threshold:
			table.append(0)
		else:
			table.append(1)
	return table


for i in os.listdir('./'):
	image=Image.open(i)
	image=image.convert('L')
	image=image.point(initTable(),'1')
	vcode=pytesseract.image_to_string(image,config='-psm 7')
	print(vcode)
	print(i)

