import segment

# set globals for this script
folder = "test_data"
pressures = ["hard"]
materials = ["paper"]

# slice up entire video into a segment for each swipe
for p in pressures:
	for m in materials:
		segment.generate_segments(folder+"/"+p+"_"+m+".mov", True)
print('All segments have been generated.')
print('Please check that all generated segments are valid.\n')
input('Press ENTER to continue...')

print('Continued')
