
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res




#Creating GUI with tkinter
import tkinter
from tkinter import *

BG_GRAY = "#27496d"
BG_COLOR = "#142850"
TEXT_COLOR = "#dae1e7"

FONT = "Roboto 14"
FONT_BOLD = "Roboto 14 bold"


class ChatApplication:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("ABC Insititute Chatbot")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=480, height=660, bg="#1b262c")

        # head label
        head_label = Label(self.window, bg="#1b262c", fg="#bbe1fa", text="Welcome to the ABC Institute Chatbot", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = Label(self.window, width=450, bg="#bbe1fa")
        line.place(relwidth=1, rely=0.07, relheight=0.1)

        # text widget
        self.text_widget = Text(self.window, width=20, height=2, bg="#1b262c", fg="#bbe1fa",font=FONT, padx=10, pady=15, bd=0)
        self.text_widget.place(relheight=0.90, relwidth=1, rely=0.075)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # scroll bar
        scrollbar = Scrollbar(self.text_widget, activebackground="#3282b8", width=10, bg="#0f4c75")
        scrollbar.place(relheight=1, relx=1)
        scrollbar.configure(command=self.text_widget.yview)

        # bottom label
        bottom_label = Label(self.window, bg="#1b262c", height=40)
        bottom_label.place(relwidth=1, rely=0.9)

        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#bbe1fa", fg="#0f4c75", font=FONT, relief=FLAT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.015)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, relief=RAISED, width=20, bg="#0f4c75", fg="#bbe1fa", command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")

    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        msg2 = f"ABC Bot : {chatbot_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)


if __name__ == "__main__":
    app = ChatApplication()
    app.run()

# def send():
#     msg = EntryBox.get("1.0",'end-1c').strip()
#     EntryBox.delete("0.0",END)
# 
#     ChatLog.config(state=NORMAL)
#     ChatLog.insert(END, "You: " + msg + '\n\n')
#     ChatLog.config(foreground="#0077fd", font=("Roboto", 14))
# 
#     res = chatbot_response(msg)
#     ChatLog.insert(END, "ABC Bot: " + res + '\n\n')
# 
#     ChatLog.config(state=DISABLED)
#     ChatLog.yview(END)
# 
# 
# base = Tk()
# base.title("ABC Campus Chat Bot")
# base.configure(bg="#2c3e50")
# base.geometry("440x590")
# base.resizable(width=FALSE, height=FALSE)
# 
# 
# 
# #Create Chat window
# ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Roboto", fg="black")
# 
# ChatLog.config(state=DISABLED)
# 
# #Bind scrollbar to Chat window
# scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
# ChatLog['yscrollcommand'] = scrollbar.set
# 
# #Create Button to send message
# SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="10", height=5,
#                     bd=0, bg="#0077fd", activebackground="#3d82e9",fg='black',
#                     command= send )
# 
# #Create the box to enter message
# EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial", fg='black')
# # EntryBox.bind("<Return>", send)
# 
# 
# 
# 
# #Place all components on the screen
# scrollbar.place(x=415,y=6, height=530)
# ChatLog.place(x=6,y=6, height=530, width=420)
# EntryBox.place(x=6, y=542, height=40, width=310)
# SendButton.place(x=320, y=542, height=40)
# 
# base.mainloop()
