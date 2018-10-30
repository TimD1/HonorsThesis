# Thermal Finger Swipe Pressure Detection

### Procedure
0. __ffmpeg__ was used to crop the videos in time to avoid dynamic rescaling occurring at the start of videos. The syntax for performing this with minimal information loss due to compression is as follows:
ffmpeg -i input.mov -ss start/fps -t (end-start)/fps -vf "crop=256:256:xstart:ystart" -c:v libx264 -crf 17 output.mov
1. __segment.py__ is called manually for each unsegmented video with the correct x and y parameters to capture swipe paths.
2. __main.py__ Runs entire pipeline, using globals _folder_, _users_, _pressures_, and _materials_. A second input argument ("segments", "frames") can specify where on the pipeline to start.


### Scripts
1. __main.py__ Runs entire pipeline, using globals _folder_, _users_, _pressures_, and _materials_. A second input argument ("segments", "frames") can specify where on the pipeline to start.
2. __segment.py__ segments a single long video capture into consecutive swipes



