import segment
import imageio
import scipy.misc
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum

# set globals for this script
folder = "data"
users = ["alex"]
pressures = ["soft", "hard"]
materials = ["concrete"]

# define various entry points to script
class StartAt(IntEnum):
	SEGMENTS = 1
	FRAMES = 2
start = StartAt.SEGMENTS

# validate command-line parameters
if len(sys.argv) == 1:
	pass
elif len(sys.argv) == 2:
	if sys.argv[1] == "segments":
		start = StartAt.SEGMENTS
	elif sys.argv[1] == "frames":
		start= StartAt.FRAMES
	else:
		print('Usage: \n\tpython main.py\n\tpython main.py <start>\n\t\twhere <start> is one of: "segments", "frames"')
		sys.exit(1)
else:
	print('Usage: \n\tpython main.py\n\tpython main.py <start>\n\t\twhere <start> is one of: "segments", "frames"')
	sys.exit(1)


if start <= StartAt.SEGMENTS:
	# clear directory "user/frames/" which stores extracted frames
	print('Clearing "frames/" directories...')
	for u in users:
		if not os.path.exists(folder+"/"+u+"/frames/"):
			os.makedirs(folder+"/"+u+"/frames/")

		for f in os.listdir(folder+"/"+u+"/frames/"):
			os.remove(folder+"/"+u+"/frames/"+f)
	print('All frames/ directories cleared.')

	# generate profile graph of each swipe, to determine which frames to select
	print('Generating time-intensity profiles for each swipe...')
	for u in users:
		for f in os.listdir(folder+"/"+u+"/segments/"):
			x = []
			y1 = []
			y2 = []
			filename = folder + "/" + u + "/segments/" + f
			reader = imageio.get_reader(filename, 'ffmpeg')
			for i, image in enumerate(reader):
				image = np.array(image).astype(np.uint8)[:,:,0]
				sum1 = np.sum(image)
				image[image == 0] = 1
				sum2 = np.sum(np.log(image))
				x.append(i)
				y1.append(sum1)
				y2.append(sum2)
				scipy.misc.toimage(image).save(folder+"/"+u+"/frames/"+
						os.path.splitext(f)[0]+"_"+str(i)+".jpg")

			plt.plot(x,y1,'b')
			plt.title('Swipe ' + f + ' sum')
			plt.show()
			plt.plot(x,y2,'g')
			plt.title('Swipe ' + f + ' logsum')
			plt.show()
	print('All time-intensity profiles have been generated.')


# # generate images for each swipe
# for u in users:
# 	# create folder to store select frames
# 	for f in os.listdir(folder+"/"+u+"/segments/"):
		

print('End')
