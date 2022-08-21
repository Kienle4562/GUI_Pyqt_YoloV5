import os
import xml.etree.ElementTree as ET
classid={'back':0,'circle':1,'clap':2,'forward':3,'lcircle':4,'wave':5}  #类别列表，与训练配置文件中的顺序保持一直
annotation='./标签2/'    #xml所在的文件
savepath='./labels/'                   #写好的txt放在labels下的train和val
for xmlfile in ('train/','val/'):
	file=os.listdir(annotation+xmlfile)
	for i in file:
		infile=annotation+xmlfile+i
		outfile=open(savepath+xmlfile+i[:-4]+'.txt','w')
		tree=ET.parse(infile)
		root=tree.getroot()
		size=root.find('size')
		w_image=float(size.find('width').text)
		h_image=float(size.find('height').text)
		for obj in root.iter('object'):
			classname=obj.find('name').text
			if(classname != 'back' and classname != 'circle'and classname != 'clap'and classname != 'forward'and classname != 'lcircle'and classname != 'wave'):
				print("出错位置:"+i+"\n出错标签:"+classname)

			cls_id=classid[classname]
			xmlbox=obj.find('bndbox')
			xmin=float(xmlbox.find('xmin').text)
			xmax=float(xmlbox.find('xmax').text)
			ymin=float(xmlbox.find('ymin').text)
			ymax=float(xmlbox.find('ymax').text)
			x_center=((xmin+xmax)/2-1)/w_image
			y_center=((ymin+ymax)/2-1)/h_image
			w=(xmax-xmin)/w_image
			h=(ymax-ymin)/h_image
			outfile.write(str(cls_id)+" "+str(x_center)+" "+str(y_center)+" "+str(w)+" "+str(h)+'\n')

		outfile.close()
