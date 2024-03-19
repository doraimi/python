#
#can work , but slowly(maybe I used CPU) , and not smoothly repsonse
#

# for speech-to-text
import speech_recognition as sr
# for text-to-speech
from gtts import gTTS
# for language model
import transformers
import os
import time
# for data
import os
import datetime
import numpy as np
# Building the AI
class ChatBot():
    def __init__(self, name):
        print("----- Starting up", name, "-----")
        self.name = name
    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            print("Listening...")
            audio = recognizer.listen(mic)
            self.text="ERROR"
        try:
            self.text = recognizer.recognize_google(audio)
            print("Me  --> ", self.text)
        except:
            print("Me  -->  ERROR")
    @staticmethod
    def text_to_speech(text):
        print("Dev --> ", text)
        #Here, we will be using GTTS or Google Text to Speech library to save mp3 files on the file system 
        #which can be easily played back.
        mp3FileName="res-" + time.strftime("%Y%m%d-%H%M%S") +".mp3"
        speaker = gTTS(text=text, lang="en", slow=False)
        #speaker.save("res.mp3")
        speaker.save(mp3FileName)
        #statbuf = os.stat("res.mp3")
        statbuf = os.stat(mp3FileName)
        mbytes = statbuf.st_size / 1024
        duration = mbytes / 200
        #os.system('start res.mp3')  #if you are using mac->afplay or else for windows->start
        os.system('start '+mp3FileName)  #if you are using mac->afplay or else for windows->start
        # os.system("close res.mp3")
        time.sleep(int(50*duration))
        #os.remove("res.mp3")
    def wake_up(self, text):
        ### When you say “Hey Dev” or “Hello Dev” the bot will become active.
        return True if self.name in text.lower() else False
    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')
# Running the AI
if __name__ == "__main__":
    # the chatbot named "dev"
    #ai = ChatBot(name="dev")
    ai = ChatBot(name="emma")
    # In the code below, we have specifically used the DialogGPT AI chatbot, 
    # trained and created by Microsoft based on millions of conversations and ongoing chats on the Reddit platform in a given time.
    nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")

    #Time to try it out
    #input_text = "hello!"
    #nlp(transformers.Conversation(input_text), pad_token_id=50256)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    ex=True
    while ex:
        ai.speech_to_text()
        ## wake up
        if ai.wake_up(ai.text) is True:
            res = "Hello I am Dave the AI, what can I do for you?"
        ## action time
        elif "time" in ai.text:
            res = ai.action_time()
        ## respond politely
        elif any(i in ai.text for i in ["thank","thanks"]):
            res = np.random.choice(["you're welcome!","anytime!","no problem!","cool!","I'm here if you need me!","mention not"])
        elif any(i in ai.text for i in ["exit","close"]):
            res = np.random.choice(["Tata","Have a good day","Bye","Goodbye","Hope to meet soon","peace out!"])
            ex=False
        ## conversation
        else:   
            if ai.text=="ERROR":
                res="Sorry, come again?"
            else:
                chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256)
                res = str(chat)
                res = res[res.find("bot >> ")+6:].strip()
        ai.text_to_speech(res)
    print("----- Closing down Dev -----")