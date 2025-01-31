from transcriber import getTranscript
from semantic import find_similarity

def main() :
    fileLink = "https://merls-audio.s3.us-east-2.amazonaws.com/11-27-2024/user_rebecca/question_1_1732680999927.webm"
    referenceSentence = "Tom and his brothers like to eat all the candy."
    #en for english zh for chinese
    result = getTranscript(fileLink, "en")
    if (result == "") :
        print("error in transcribing")
        return

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

if __name__ == "__main__":
    main()