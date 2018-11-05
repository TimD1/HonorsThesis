import segment
import imageio
import scipy.misc
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
from operator import itemgetter

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
if len(sys.argv) == 2:
	if sys.argv[1] == "segments":
		start = StartAt.SEGMENTS
	elif sys.argv[1] == "frames":
		start= StartAt.FRAMES
	else:
		print('Usage: \n\tpython main.py <start>')
		print('\t\twhere <start> is one of: "segments", "frames"')
		sys.exit(1)
else:
	print('Usage: \n\tpython main.py <start>')
	print('\t\twhere <start> is one of: "segments", "frames"')
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

	# determine which frames to use for further analysis
	print('Extracting relevant frames from each swipe video...')
	for u in users:
		for f in os.listdir(folder+"/"+u+"/segments/"):

			# calculate sum(log(frame_vals)) and keep top 25 frames
			frame_sums = []
			filename = folder + "/" + u + "/segments/" + f
			reader = imageio.get_reader(filename, 'ffmpeg')
			for i, image in enumerate(reader):
				image = np.array(image).astype(np.float32)[:,:,0]
				image[image < 1] += 1
				total = np.sum(np.log2(image))
				frame_sums.append((i, total))
			frame_sums.sort(key=itemgetter(1), reverse=True)
			frames = [x for (x,y) in frame_sums[:25]]
			for i, image in enumerate(reader):
				image = np.array(image).astype(np.uint8)[:,:,0]
				if i not in frames:
					continue
				else:
					# print(filename, i, np.any(np.isnan(
					scipy.misc.toimage(image).save(folder+"/"+u+"/frames/"+
							os.path.splitext(f)[0]+"_"+str(i)+".jpg")
	print('25 frames have been extracted from each video..')


# # generate images for each swipe
# for u in users:
# 	# create folder to store select frames
# 	for f in os.listdir(folder+"/"+u+"/segments/"):
		

print('End')
