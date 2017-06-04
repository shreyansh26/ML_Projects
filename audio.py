import cmu_sphinx4

audioURL = 'http://www.wavsource.com/snds_2017-05-21_1278357624936861/movies/spidey/enhanced.wav'

transcriber = cmu_sphinx4.Transcriber(audioURL)

for line in transcriber.transcript_stream():
	print(line)
a=input()