#---------------------------------IMPORTS-----------------------------------------------------------#

from flask import Flask, jsonify, request, url_for
from openai import AzureOpenAI
import numpy as np
import pdfplumber
import xml.etree.ElementTree as ET
import re
import faiss 
from transformers import AutoTokenizer, AutoModel
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
import threading
import pickle
import torch
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#---------------------------------FLASK APP INITIALIZATION-----------------------------------------------------------#

app = Flask("chat~bot")
app.secret_key = "OcmYJ2iYixJuagMgc8YHKAurXSPFOBax4qcnJKPkX5UYEdWStWRMUnIJFsaGcbOs"

#---------------------------------GPT 4 INITIALIZATION-----------------------------------------------------------#

client = AzureOpenAI(
    azure_endpoint="https://gpt4-49.openai.azure.com/",
    api_key="6070caa88aa34b2b88fafab183df694b",
    api_version="2024-02-15-preview"
)

#---------------------------------CREATING GLOBAL INDICES-----------------------------------------------------------#

index = None
all_text_chunks = []


#---------------------------------FUNCTIONS FOR SAVING AND LOADING DATA-----------------------------------------------------------#


def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        

def save_faiss_index(index, filename='faiss_index.faiss'):
    faiss.write_index(index, filename)

def load_faiss_index(filename='faiss_index.faiss'):
    try:
        return faiss.read_index(filename)
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
        return None

def save_text_chunks(text_chunks, filename='text_chunks.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(text_chunks, f)

def load_text_chunks(filename='text_chunks.pkl'):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Text chunks file not found.")
        return []




#---------------------------------PROCESSING PDF FILES-----------------------------------------------------------#
#---------------------------------CONVERTING PDF FILES TO XML-----------------------------------------------------------#

def sanitize_for_xml(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)
    return text

def pdf_to_xml(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages
        root = ET.Element("document")
        for i, page in enumerate(pages):
            page_text = page.extract_text(x_tolerance=2, y_tolerance=0)
            if page_text:
                page_text = sanitize_for_xml(page_text)
                page_element = ET.SubElement(root, "page", number=str(i + 1))
                text_element = ET.SubElement(page_element, "text")
                text_element.text = page_text
            for table in page.extract_tables():
                table_element = ET.SubElement(page_element, "table")
                for row in table:
                    row_element = ET.SubElement(table_element, "row")
                    for cell in row:
                        cell_text = sanitize_for_xml(cell) if cell else ""
                        cell_element = ET.SubElement(row_element, "cell")
                        cell_element.text = cell_text
        xml_path = f"{pdf_path}.xml"
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    return xml_path

#---------------------------------EXTRACTING TEXT FROM XML-----------------------------------------------------------#


def extract_text_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    texts = []
    for page in root.findall(".//page"):
        page_texts = [node.text for node in page.findall(".//text") if node.text]
        # Extracting text from tables
        for table in page.findall(".//table"):
            for row in table.findall(".//row"):
                row_texts = [cell.text.strip() for cell in row.findall(".//cell") if cell.text and cell.text.strip()]
                if row_texts:
                    texts.append(' | '.join(row_texts))  # Use a delimiter to separate cell texts
        page_text = ' '.join(page_texts)
        texts.append(page_text)
    return " ".join(texts)


#---------------------------------CHUNKING TEXT-----------------------------------------------------------#





def chunk_text(text, max_tokens=300, overlap=50, model_dir="../llama2/llama2-chat-7b"):
    
    
    # Validate parameters
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_tokens:
        raise ValueError("overlap must be less than max_tokens")
    
    
    
    # Load the Llama2 tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    
    # Tokenize the text: convert text to a list of token ids
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    # Initialize variables for chunking
    chunks = []
    index = 0  # start index of the next chunk
    
    while index < len(token_ids):
        # Determine the segment of token ids to decode
        chunk_ids = token_ids[index:index + max_tokens]
        # Convert token ids back to text
        chunk_text = tokenizer.decode(chunk_ids, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
        #print(chunks)
        
        
        # Update the start index by moving max_tokens forward and stepping back by overlap
        index += (max_tokens - overlap)
    
    return chunks


#---------------------------------SUMMARIZING TEXT CHUNKS-----------------------------------------------------------#

#summarizing the chunks
def summarize_text_chunks(text_chunks):
    summarized_texts = []
    for chunk in text_chunks:
        
        message_text_summarize = [       
        {"role": "system", "content": "Summarize the given text."},
        {"role": "user", "content": chunk}
        ]
        
        if len(chunk) > 100:  # Ensure there's enough text to summarize
            
            completion = client.chat.completions.create(
                model="gpt4-49", 
                messages = message_text_summarize,
                max_tokens=200,
                temperature=0.4,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            summary = completion.choices[0].message.content if completion.choices else "Summary not generated."
        else:
            summary = chunk
        summarized_texts.append(summary)
    # print("under summarize_text_chunks")
    # print(summarized_texts)
    # exit()
    return summarized_texts



#---------------------------------VECTOR NORMALIZATION FUNCTION-----------------------------------------------------------#

# def normalize_vectors(vectors):
#     norms = np.linalg.norm(vectors, axis=1, keepdims=True)
#     return vectors / norms




#---------------------------------LOADING LLAMA 2-----------------------------------------------------------#

model_dir = "../llama2/llama2-chat-7b"
model = LlamaForCausalLM.from_pretrained(model_dir)
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token



#---------------------------------GENERATING EMBEDDINGS-----------------------------------------------------------#

# Load the Sentence Transformer model
#model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")

def vectorize_text(text_chunks):
    # Generate embeddings for the given text chunks
    model_path = 'Alibaba-NLP/gte-large-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    # Tokenize the input texts
    batch_dict = tokenizer(text_chunks, max_length=8192, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    # embeddings = outputs.last_hidden_state[:, 0].detach()
    embeddings = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
    # (Optionally) normalize embeddings
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    
    embeddings_np = embeddings.detach().numpy()
    all_embeddings_np = np.vstack(embeddings_np)

    return all_embeddings_np
#---------------------------------CREATING FAISS INDEX-----------------------------------------------------------#


def create_update_index(embeddings, index_file, summaries_file, summaries):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_file)
    #print(summaries)
    save_data(summaries, summaries_file)
    print(f"Updated FAISS index and summaries with dimension {dimension} and {len(summaries)} summaries..")


#---------------------------------INTIALIZING INDICES-----------------------------------------------------------#

def initialize_index():
    global index, all_text_chunks
    index = load_faiss_index()
    all_text_chunks = load_text_chunks()

    if index is None or all_text_chunks == []:
        print("No index or text chunks found. Generating new.")
        all_text_chunks, all_embeddings = process_pdfs_in_directory("static/pdfs")
        if all_embeddings.size > 0:
            index = faiss.IndexFlatIP(all_embeddings.shape[1])
            index.add(all_embeddings)
            save_faiss_index(index)
            save_text_chunks(all_text_chunks)
            print("New index and text chunks created and saved.")
        else:
            print("Failed to generate embeddings. Check your PDFs and model.")


#---------------------------------PROCESSING DIRECTORY----------------------------------------------------------#
 

def process_pdfs_in_directory(directory_path="static/pdfs"):
    all_text_chunks = []
    all_summaries = []
    all_embeddings = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            xml_path = pdf_to_xml(pdf_path)
            #xml_path = f"{pdf_path}.xml"
            print(f"Processing {pdf_path}...")
            text = extract_text_from_xml(xml_path)
            chunks = chunk_text(text)
            print("chunks created")
            #print(chunks)
            summaries = summarize_text_chunks(chunks)
            print("summarized text:" , summaries)
            
            vectors = vectorize_text(summaries)

            all_text_chunks.extend([(chunk, filename) for chunk in chunks])            
            all_summaries.extend(summaries)
            all_embeddings.extend(vectors)
            print(f"Processed {pdf_path}.")

    all_embeddings_np = np.vstack(all_embeddings)  # Convert list of np.array to a single np.array
    create_update_index(all_embeddings_np, index_file='faiss_index.faiss', summaries_file='summaries.pkl', summaries=all_summaries)
    save_data(all_text_chunks, 'text_chunks.pkl')
    print("All processing complete.")

    return all_text_chunks, all_embeddings_np

#---------------------------------APP INITIALIZATION-----------------------------------------------------------#


@app.route('/', methods=['POST'])
def home():
    global index, all_text_chunks
    query = request.json.get('query', '').strip().lower() 
    
    print(f"Received query: {query}")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    if not index or not all_text_chunks:
        return jsonify({"error": "Service not ready, try again later."}), 503
    
    
    
    # Mapping of simple queries to their responses
    simple_responses = {
        "hi": "Hello! How can I help you today?",
        "hi!": "Hello! How can I help you today?",
        "hello": "Hello! How can I help you today?",
        "hello!": "Hello! How can I help you today?",
        "who are you": "I am a chatbot created to assist you. How can I help you today?",
        "who are you?": "I am a chatbot created to assist you. How can I help you today?",
        "how are you" : "I am good, thanks for asking. How can I help you today?",
        "how are you?" : "I am good, thanks for asking. How can I help you today?",
        "what do you know": "I can give you information about IIT Gandhinagar."
    }

    # Check if the query is one of the recognized simple queries
    if query in simple_responses:
        return simple_responses[query]

    

    query_vector = vectorize_text([query])[0]
    D, I = index.search(np.array([query_vector]), 5)
    results = [(all_text_chunks[idx][0], all_text_chunks[idx][1], idx) for idx in I[0]]
    print(f"Results: {results}")
    print("distance values: ", D)
    print("indices: ", I)
    
    response_sentences = [result[0] for result in results]
    response_pdf_names = [result[1] for result in results]
    response_pdf_urls = [url_for('static', filename=f'pdfs/{name}', _external=True).lstrip('/') for name in response_pdf_names]



    sorted_results = sorted(zip(results, D[0]), key=lambda x: x[1])
    closest_result = sorted_results[0][0]  # Get the result with the smallest distance
    closest_pdf_url = url_for('static', filename=f'pdfs/{closest_result[1]}', _external=True)
    print("closest result: ", closest_result)
    print("closest index: ", closest_result[2])


    input_text = query + " " + " ".join(response_sentences)

    message_text = [       
        {"role": "system", "content": "You are an AI assistant that helps people find information by answering their queries regarding IIT Gandhinagar whose advisories you have been trained on. The input given to you is in form of a question leading to 5 similar sentences that are potential answers to the question, your task is to generate one meaningful answer by analyzing all the given answers. If the query and the given 5 sentences don't match, continue with a normal conversation and if you don't think the output is relevant to the query say 'apologies, I don't have answer to that question' and please don't give any further information. Don't mention any information about how you have been trained. Don't answer in incomplete sentences. If the query is just a greeting answer just a greeting! Answer in short. If the query is just 'hello' or 'hi' answer with 'Hello! How can I help you today?'"},
        {"role": "user", "content": input_text}
    ]
    
    completion = client.chat.completions.create(
    model="gpt4-49", 
    messages = message_text,
    temperature=0.4,
    max_tokens=200,
    top_p=0.95,
    frequency_penalty=2,
    presence_penalty=0,
    )

    if completion.choices:
        generated_text = completion.choices[0].message.content
    else:
        generated_text = "No completion found."

    print("output:\n", generated_text)
    

    # Vectorize the generated text for the second similarity search
    generated_vector = vectorize_text([generated_text])[0]

     # Prepare a sub-index from the initial top 5 results
    top_indices = [res[2] for res in results]  # Extract original global indices from results
    sub_embeddings = np.vstack([vectorize_text([text])[0] for text, _, _ in results])  # Use only the text part for re-vectorization
    sub_index = faiss.IndexFlatIP(generated_vector.shape[0])
    sub_index.add(sub_embeddings)

    # Search for the most similar result within these 5
    D_gen, I_gen = sub_index.search(np.array([generated_vector]), 3)
    print(D_gen)
    most_relevant_chunk_index = top_indices[I_gen[0][0]]  # Correct mapping from local to global index
    most_relevant_result = all_text_chunks[most_relevant_chunk_index]
    print(most_relevant_chunk_index)
    most_relevant_pdf_url = url_for('static', filename=f'pdfs/{most_relevant_result[1]}', _external=True)


    answer = generated_text + "<br><a target  = '_blank' href='" + most_relevant_pdf_url + "' ><b>Source</b></a>"   
    print("answer generated")
    
    # Check if any standard phrases are in the generated_text
    if ("Apologies" in generated_text or "No completion found" in generated_text ):
        return generated_text

    return answer
    
if __name__ == "__main__":
    threading.Thread(target=initialize_index).start()
    app.run(host='0.0.0.0', port=3000, debug=True)

