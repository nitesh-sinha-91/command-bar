
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.agents.agent_toolkits.openapi import planner
from rest_framework.decorators import api_view
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

persist_dir = 'command_bar_db'

@api_view(['GET'])
def train(request):
    # filename = os.path.join()
    # print(data)
    loader = TextLoader(os.path.join(os.getcwd(), 'information.txt'))
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings,persist_directory=persist_dir)
    db.persist()    
    return Response("Done", status=HTTP_200_OK)

@api_view(['GET'])
def get_command_bar_data(request):
    text = request.GET.get('text')
    # agent = create_csv_agent(
    #         OpenAI(temperature=0),
    #         "project_name.csv",
    #         verbose=True
    #     )
    template = f"""This text will have a relevant command. Fetch it and return it to me: "{text}". Return only one command name or NA."""
    template2 = f'This text will have a project name within it, it could be a single word or couple of words. It will not be similar to the command and mostly will not be an English word. Fetch it and return it to me: "{text}". Return only project name if you can find else return NA'
    template3 = f'Return me the text that is present in any project names and in this input "{text}". If it does not matches return NA'
  
    vector_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=OpenAIEmbeddings()
    )
    retriever = vector_db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-4", temperature=0.7), chain_type="stuff", retriever=retriever)
    res = qa.run(template)
    res2 = qa.run(template2)
    res2 = res2.replace('text is', '')        
    res2 = res2.replace('\"', '')
    if res2 == "NA":
        res2 = ""
    if "don't know" in res:
        res2 = ""
    if res == "NA" and len(res2) > 0:
        res = "GOTO-PROJECT-DETAILS"
        
    return Response({
        "command": res,
        "text": res2,
    }, status=HTTP_200_OK)
        

"""
Table contains mapping of commands and actionItemName that user can perform.
using actionItemName and description try to find for which user want to perform any action and which command he want to execute.
user may provide some text that we need to search for that actionItemName. Try to find that search text too. For example if user write 
navigate me to xyz project recc. Then GO_TO is command, xyz is search text and recce is actionItemName.
Always return ans in this format - command,actionItemName,searchText. If you can't find any relevant result then return NA.
User text is - 

template2 = 
    Tell me text that can be a relavent project name or id hidden inside this input: "{text}". Return only project name or id  if you can find else return NA.
    For Example:
    prompt: "go to snag module in test", response: "test"
    prompt: "go to boq in xyz", response: "xyz"
    prompt: "project test open snag module", response: "test"
    prompt: "project neeman hydrabad open snag module", response: "neeman hydrabad"
    prompt: "in abcdefg go to snags", response "abcdefg"
    prompt: "of chennai open design page", response: "chennai"
    prompt: "show me design of pantaloon", response: "pantaloon"

"""