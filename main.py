from transcriber import getTranscript
from semantic import find_similarity

audioFile = "file/test.webm"
referenceSentence = "Tom and his brothers like to eat all the candy."
#en for english zh for chinese
result = getTranscript(audioFile, "en")
print("transcribed: " + result)
similarity = find_similarity(result, referenceSentence)
print("cosine similarity: " + str(similarity))
if (similarity < 0.6) :
    print("Answer is completely different")
elif (similarity < 0.75) :
    print("Answer is somewhat related")
elif (similarity < 0.875) :
    print("Answer is similar")
elif (similarity <= 1) :
    print("Answer is very similar")

