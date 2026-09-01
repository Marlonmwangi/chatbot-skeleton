#import os to handle all os subfunctions like checking for path and creating new paths
import os
#os.path.exists()checks if a path exists
#os.path.join()creates the path

#import json to handle json message parsing
import json
#json.load()reads json
#json.dumps()writes json
from typing import TypedDict,Annotated
#import datetime to handle timestamps in our chatbot messages
from datetime import datetime,timezone
#datetime.now().isoformat() returns date and time in readable format
#import message types like AIMessage,HumanMessage and BaseMessage from langchain_core.messages
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage
#import stategraph(the whiteboard or what is shown on the screen)from langgraph.graph
from langgraph.graph import StateGraph
#import OllamaLLM(the model )from langchain_ollama
from langchain_ollama import OllamaLLM
#import add_messages(appends existing conversation to preserve context)from langgraph.graph.messages
from langgraph.graph.message import add_messages
#import START and END (where conversation starts and ends)from langgraph.graph
from langgraph.graph import START,END


#===============================================================
                     #INITIALIZE THE LLM
#===============================================================
print("Loading your LLM.....\n")
try:
    llm=OllamaLLM(model="llama3.2:3b",temperature=0)
    print("Model Loaded!!!\n")
except Exception as e:
    print(f"ERROR!!:{e}")
    print("Make sure Ollama is running:ollama serve")
    exit(1)

#=========================================================================================
                #MEMORY MANAGEMENT
#=========================================================================================

#create class for memory storage.Separate it from chat logic class to make it clean
class ConversationMemory():
    #call the library with self then  name the library("chat_memory.json")
    def __init__(self,memory_file="chat_memory.json"):
         self.memory_file=memory_file
         #variable to load conversations
         self.conversations=self._load_memory()
    #_load_memory()loads preexisting conversations
    def _load_memory(self):
        #check if memory file exists.os.path.exists returns True/False
        if os.path.exists(self.memory_file):
            #with open() opens file."with"prefix closes file automatically."r" opens the file in read mode.
            with open(self.memory_file,"r")as f:
                return json.load(f)#json.load()converts json text to python dict.
        return{}#return an empty dict if file doesn't exist.This is the first time running.
    
    #now write ur save  memory function 
    def _save_memory(self):
        #"w"writes a file
        with open(self.memory_file,"w")as f:
            #indent=2 saves file in json with indent of 2 for easy readability
            json.dump(self.conversations,f,indent=2)
    
    #next we will get user info so that we can load past conversations from the correct person later onwards
    def get_user_profile(self,user_id:str):#the :str specifies that inputed id should be in string form only.
        #return userID if it exists else return user profile with a dict with the message(conversation),creation date and save total conversations at 0 if its a new user.
        return self.conversations.get(user_id,{
            "messages":[],
            "creation_date":datetime.now().isoformat(),
            "total_conversations":0
        })
    
    #How to save messages . The _save_memory function just allowed us to store the entire convo as a json object.this one makes us save specific sections appropriately, i.e timestamp
    def save_message(self,user_id:str,role:str,content:str):
        '''
          Save a single message to history.
          user_id shows who the message belongs to.
          role:shows whether its written by the AI assistant or Human.
          content:exact message to be stored.
          timestamp:time it was written.
        '''
        #create user profile if user does not exist
        if user_id not in self.conversations:
            self.conversations[user_id]={
                "messages":[],#messages is a list of message
                "creation_date":datetime.now().isoformat(),
                "total_conversations":0
            }
        #create a message structure
        message={
            "role":role,
            "content":content,
            "timestamp":datetime.now().isoformat()
                }
        #now append each message to messages
        self.conversations[user_id]["messages"].append(message)
        #save to memory immediately so as not to lose data
        self._save_memory()

    def get_relevant_context(self,user_id:str,limit:int=5)->str:
        '''
           Get previous messages to help LLM understand the context and keep up with the conversation well

           user_id:from whom are we getting the messages from.
           limit:helps extract the last no of  message so as to provide as solid context of what is going on and not mix up topics.


        '''
        profile=self.get_user_profile(user_id)
        messages=profile.get("messages",[])#gets the messages as a list.
        recent_messages=messages[-limit:]if messages else[]
        if not recent_messages:
            return "No previous Conversation history!"
        context="Previous conversation history:\n"
        #extract role(human or assistant) from the messages so that the LLM might understand
        for msg in recent_messages:
            role="User"if msg["role"]=="human"else "Assistant"
            context +=f"{role}:{msg['content']}\n"
        return context

        #=========================================================================================================================================================================
                                 #STATE MANAGEMENT

        #=========================================================================================================================================================================

class ChatState(TypedDict):
        '''
                       This defines what information the chatbot keeps track of.
                       TypedDict is just a blueprint of keys and value types.
        
        '''
        messages:Annotated[list[BaseMessage],add_messages]
        user_id:str
        system_prompt:str
        previous_context:str

#========================================================================================================
                   #MAIN CHATBOT LOGIC
#========================================================================================================
def chat_node(state:ChatState)->dict:
    '''
       This function is called on every run.
       state:ChatState refers to the current conversation state.
       The LLM processes information from previous chat messages and context when this function runs.
       It returns a dict of responses
    '''
    print(f"Processing run:{len(state['messages'])//2 +1}....")
    #Buld the system message that contains the system prompt and context that tells the llm how to behave.
    system_message=f"""
                       {state['system_prompt']}{state['previous_context']}
                    Remember:Be kind and respectful.Always remember facts about the user.

                    """
    #Create a full message list including the system prompt
    messages_to_send=[HumanMessage(content=system_message)]+state['messages']
    #Send to LLM and get response
    response=llm.invoke(messages_to_send)
    print(f" Response:{response[:80]}....")
    #Return response to be added to state
    return{
         'messages': AIMessage(content=response.content if hasattr(response, 'content') else str(response))
    }

#create the graph
def create_graph():
    """
    Create the langgraph that orchestrates the conversation.
    Think of it as a building the conversation pipeline.

    Returns:
    The compiled graph ready to run.
    """
    #Create the StateGraph(whiteboard/screen)    
    graph=StateGraph(ChatState)

    #Add our chat node to the graph
    graph.add_node("chat_node",chat_node)
    #Connect START to the chat node(conversation start point)
    graph.add_edge(START,"chat_node")
    #Connect END to the chat node(conversation ends here)
    graph.add_edge("chat_node",END)

    #compile the graph into an excecutable form
    return graph.compile()

class MultiConversationalChatbot:
    '''
    Brings all parts together(memory+graph+logic)


    '''
    def __init__(self):#initializes memory and graph.
                self.memory=ConversationMemory()#create a memory system.We can then access the memory easily using self.memory.
                self.graph=create_graph()#Ready to invoke for conversations.
                self.user_id=None#Starts with no user.Will be set when user signs in.
    
    def get_system_prompt(self)->str:
        '''
                    What: Return instructions for LLM
                    Why: Controls chatbot personality and behavior
                    How: Return a string with detailed instructions
                    
        '''
        return """You are a helpful, friendly chatbot assistant.
You have access to previous conversations with this user.
- Be personable and remember what the user has told you
- Answer questions helpfully and accurately
- If you don't know something, say so honestly
- Keep responses concise unless asked for more detail"""

    def set_user(self,user_id:str):
        """
        Sets what user we are talkin about.
        This loads their profile and previous conversations.
        
        """
        self.user_id=user_id
        profile=self.memory.get_user_profile(user_id)

        #Count how many times this user has chatted before.
        num_conversations=len(profile.get("messages",[]))//2#Divide by 2 because each turn=2messages
        print(f"\n User:{user_id}")
        print(f"Number of conversations:{num_conversations}")
        print(f"Member since:{profile.get('creation_date','New_user')}\n")

    def chat(self,user_message:str)->str:
        """
        Have a single turn of conversation.
        Return(str):what the chatbot said
        
        """
        #Save user's message to memory.
        self.memory.save_message(self.user_id,"human",user_message)

        #get context from previous conversations
        previous_context=self.memory.get_relevant_context(self.user_id)
        #Create initial state for this turn
        initial_state={
            "messages":[HumanMessage(content=user_message)],
            "user_id":self.user_id,
            "system_prompt":self.get_system_prompt(),
            "previous_context":previous_context

        }

        #run the graph(invoke sends the state through the graph)
        result=self.graph.invoke(initial_state)
        #Extract the response
        response=result['messages'][-1].content
        #Save assistant's response to memory
        self.memory.save_message(self.user_id,"assistant",response)

        return response
    def show_memory(self):
        """Show the user their conversation history"""
        profile=self.memory.get_user_profile(self.user_id)
        messages=profile.get("messages",[])

        print(f"CONVERSATION HISTORY -{self.user_id}")

        if not messages:
            print("No conversations yet!!")
            return
        #Show all messages
        for i,msg in enumerate(messages,1):
            role="You" if msg["role"]=="human" else "BOT"
            print(f"{i}.{role}:{msg['content']}\n")


#============================================================================================
                       #:MAIN EXECUTION -Run the chatbot
#============================================================================================

if __name__=="__main__":
    print("MULTICONVERSATIONAL CHATBOT WITH MEMORY")
    print()


    #create chatbot instance
    chatbot=MultiConversationalChatbot()
    #get user ID
    user_id=input("Enter your name(or ID):").strip()#strip()removes spaces at the beginning and end.
    if not user_id:
        user_id="guest"

    #Set the user
    chatbot.set_user(user_id)
    print("Commands:")
    print("-Type your message to chat")
    print("-Type 'history' to see past cpnversations.")
    print("-Type 'quit' to exit\n")
    print("--------------------------------------------------------------------------------------------------------------------------------")

    #Keep chatting till user quits!!
    while True:
        user_input=input("\n You:").strip()

        #Handles special commands
        if user_input.lower()=="quit":
            print("\n Goodbye! Your conversation has been saved.\n")
            break

        if user_input.lower()=="history":
            chatbot.show_memory()
            continue

        if not user_input:
            continue

        #Get a response from chatbot 
        try:
            response=chatbot.chat(user_input)
            print(f"\n Bot:{response}")
        except Exception as e:
            print(f"Error:{e}")
            print("Make sure Ollama is Running!")






         