from ast import Break, Global
import streamlit as st
import pyttsx3
import speech_recognition as sr
from pdf2image import convert_from_bytes
import pytesseract
import pandas as pd
from PIL import Image 
import joblib
import torch
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from transformers import BertForQuestionAnswering, AutoTokenizer
import sys


def local_vars_name_size():
	local_vars = list(locals().items())
	for var, obj in local_vars:
		print(var, sys.getsizeof(obj))

def saving_model_tokenizer():
	modelname = 'deepset/bert-base-cased-squad2'
	model = BertForQuestionAnswering.from_pretrained(modelname)
	tokenizer = AutoTokenizer.from_pretrained(modelname)
	joblib.dump(model, 'bert_model.joblib')
	joblib.dump(tokenizer, 'bert_token.joblib')

# saving_model_tokenizer()


def read_text(text):
	article = text.split(". ")
	sentences = []

	for sentence in article:
		print(sentence)
		sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
	sentences.pop() 
	
	return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
	if stopwords is None:
		stopwords = []
 
	sent1 = [w.lower() for w in sent1]
	sent2 = [w.lower() for w in sent2]
 
	all_words = list(set(sent1 + sent2))
 
	vector1 = [0] * len(all_words)
	vector2 = [0] * len(all_words)
 
	for w in sent1:
		if w in stopwords:
			continue
		vector1[all_words.index(w)] += 1
 
	for w in sent2:
		if w in stopwords:
			continue
		vector2[all_words.index(w)] += 1
 
	return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
	similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
	for idx1 in range(len(sentences)):
		for idx2 in range(len(sentences)):
			if idx1 == idx2: #ignore if both are same sentences
				continue 
			similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

	return similarity_matrix

def generate_summary(page_text, top_n=5):

	stop_words = stopwords.words('english')
	summarize_text = []

	sentences =  read_text(page_text)
	print("read_text done")
	sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
	print("build_sim_mit done")

	sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
	print("build_sentence graph done")
	scores = nx.pagerank(sentence_similarity_graph , max_iter=250)

	ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)   
	print("ranked sentence done")
	for i in range(top_n):
		summarize_text.append(" ".join(ranked_sentence[i][1]))
		final_text = str(". ".join(summarize_text))
	return final_text
	 


def Process_image(image_file):
	img = Image.open(image_file)
	ocr_text = pytesseract.image_to_string(img)
	return ocr_text

def process_pdf(file):
	# print(file)
	images = convert_from_bytes(file.read())
	print("Total No. of pages = " , len(images))
	ocr_text = ''
	page_summ = ''
	counter = 1
	if len(images) >=4:
		print("file has many pages, so summarising the text page wise.")
		for img in images:
			print("=========================================================================Processing page Number: - ", counter)
			counter += 1
			ocr_text = pytesseract.image_to_string(img)
			page_summ = generate_summary(ocr_text)      # User defined function
			page_summ += page_summ
		return page_summ
	else:
		for img in images:
			print('image' + str(counter) + '.jpg')
			counter += 1
			ocr_text = pytesseract.image_to_string(img)
			ocr_text = " " + ocr_text
		return ocr_text



def record_speech():
	r = sr.Recognizer()
	with sr.Microphone() as source:
		while True:
			bot_speak('Speak after a little pause:')
			audio = r.listen(source)
			
			try:
				text = r.recognize_google(audio)
				print("You said : {}".format(text))
				return text
			except:
				bot_speak("Sorry, i could not recognize what you said")
				bot_speak("try speaking again")

def bot_speak(context):
	engine= pyttsx3.init('sapi5')
	voices = engine.getProperty("voices")
	print(voices)
	engine.setProperty("voice",voices[1].id)

	engine.say(context)
	engine.runAndWait()

def predict_process(question_asked,text_data):
	model = joblib.load('bert_model.joblib')
	tokenizer = joblib.load('bert_token.joblib')

	input_ids = tokenizer.encode(question_asked, text_data ,truncation=True)
	tokens = tokenizer.convert_ids_to_tokens(input_ids)
	
	sep_idx = input_ids.index(tokenizer.sep_token_id)

	num_seg_a = sep_idx+1

	print("Number of tokens in the question: ", num_seg_a)

	num_seg_b = len(input_ids) - num_seg_a
	print("Number of tokens in context ", num_seg_b)

	segment_ids = [0]*num_seg_a + [1]*num_seg_b

	output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

	answer_start = torch.argmax(output.start_logits)
	answer_end = torch.argmax(output.end_logits)
	print("start and end token" , answer_start , answer_end)
	if answer_end >= answer_start:
		answer = " ".join(tokens[answer_start:answer_end+1])
		return answer
	else:
		print("unable to find the answer to this question. Please ask another question")
		return None


def main():
	st.title("Please Upload a File")

	menu = ["DocumentFiles", "Image"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Image":
		st.subheader("Image")
		image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
		if st.button("Process"):
			if image_file is not None:
				file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
				st.write(file_details)

				text_data = Process_image(image_file)  # User defined function
				st.write(text_data)

				question_asked = record_speech()    # User defined function

				st.write("Your question was: ", question_asked + "?")

				bot_speak("your question was, " + question_asked)   # User defined function

				answer = predict_process(question_asked=question_asked , text_data=text_data)  # User defined function (hssh)

				print(answer)

				if answer == None:
					bot_speak("unable to find the answer to this question, please ask another question")
				else:
					st.write("THE ANSWER IS: " , answer)
					bot_speak("the answer is, " + answer)
			# 	local_vars_name_size()     # User defined function
							
	elif choice == "DocumentFiles":
		st.subheader("DocumentFiles")
		docx_file = st.file_uploader("Upload Pdf",type=['txt','pdf'])
		if st.button("Process"):
			if docx_file is not None:
				file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
				st.write(file_details)

				# Check File Type
				if docx_file.type == "text/plain":
					text_data = str(docx_file.read(),"utf-8")
					# print(text_data)
					st.write(text_data)

					question_asked = record_speech() # User defined function

					st.write("Your question was: ", question_asked +"?")
					bot_speak("your question was, " + question_asked)   # User defined function

					answer = predict_process(question_asked=question_asked , text_data=text_data)

					print(answer)

					if answer == None:
						bot_speak("unable to find the answer to this question, please ask another question")
					else:
						st.write("THE ANSWER IS: " , answer)
						bot_speak("the answer is, " + answer)
					
					# local_vars_name_size()	

				elif docx_file.type == "application/pdf":

					print("This is the uploaded doc :==============================" , docx_file)

					text_data = process_pdf(docx_file)   # User defined function
					st.write(text_data)

					question_asked = record_speech()   # User defined function

					st.write("Your question was: ", question_asked + "?")
					bot_speak("your question was, " + question_asked)   # User defined function

					answer = predict_process(question_asked=question_asked , text_data=text_data)   # User defined function
					print(answer)

					if answer == None:
						bot_speak("unable to find the answer to this question, please ask another question")
					else:
						st.write("THE ANSWER IS:  -  " , answer)
						bot_speak("the answer is, " + answer)
					# local_vars_name_size()

if __name__ == '__main__':
	main()