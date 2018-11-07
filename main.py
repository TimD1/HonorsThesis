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
users = ["alex", "ben", "miao", "natasha", "nick", "sarah", "sean", "spencer", "tim", "yijun"]

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
		print('\tExtracting frames for user', u)
		for f in os.listdir(folder+"/"+u+"/segments/"):

			# set up image reader
			filename = folder + "/" + u + "/segments/" + f
			reader = imageio.get_reader(filename, 'ffmpeg')

			# calculate total frames, weighted/unweighted frame intensities
			total_intensity = 0
			weighted_total_intensity = 0
			frame_count = 0
			for i, image in enumerate(reader):
				frame_count += 1
				image = np.array(image).astype(np.uint8)[:,:,0]
				total_intensity += np.sum(image)
				weighted_total_intensity += (i+1)*np.sum(image)

			# select up to 25 frames in right time span with min intensity
			center_frame = weighted_total_intensity // total_intensity
			avg_intensity = total_intensity // frame_count
			for i, image in enumerate(reader):
				image = np.array(image).astype(np.uint8)[:,:,0]
				if (i >= center_frame-5 and i < center_frame + 20 and 
						np.sum(image) > 0.5*avg_intensity):
					scipy.misc.toimage(image).save(folder+"/"+u+"/frames/"+
							os.path.splitext(f)[0]+"_"+str(i)+".jpg")
	print('Up to 25 frames have been extracted from each video..')
