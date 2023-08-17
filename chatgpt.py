import os
import sys
import mad
import time

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

from mutagen.mp3 import MP3


from voiceTTS import recognize_speech

from gtts import gTTS
import pygame

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False


query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
  loader = DirectoryLoader("data/")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
conversationNo = 0
while True:
  pygame.mixer.init()
  conversationNo += 1
  if not query:
    print("Prompt: ")
    query = recognize_speech()
  if query in ['quit', 'q', 'exit']:
    pygame.mixer.music.load("exitConvo.mp3")
    pygame.mixer.music.play()
    time.sleep(2)
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])
  text = result['answer']

  tts = gTTS(text)
  tts.save("output{}.mp3".format(conversationNo))

  pygame.mixer.music.load("output{}.mp3".format(conversationNo))
  pygame.mixer.music.play()

  chat_history.append((query, result['answer']))
  query = None
  
  audio = MP3("output{}.mp3".format(conversationNo))
  duration_in_seconds = audio.info.length
  
  time.sleep(duration_in_seconds + 1)

