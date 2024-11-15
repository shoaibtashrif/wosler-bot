from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import speech_recognition as sr
import pyttsx3
from langchain.llms.base import LLM
from groq import Groq
from typing import Optional
import logging

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Custom LLM class for Groq
class GroqLLM(LLM):
    client: Groq
    model: str = "gemma2-9b-it"

    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        logger.info("🤖 Sending request to Groq")
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
        )
        return chat_completion.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "groq"

# Initialize components
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)
llm = GroqLLM(client=groq_client)
r = sr.Recognizer()
engine = pyttsx3.init()
logger.info("🚀 System initialized")

# Define the state type
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    chatbot_response: str

# Create the graph
graph_builder = StateGraph(State)

def listen(state: State):
    """Convert audio input to text using speech_recognition."""
    logger.info("👂 Listening...")
    with sr.Microphone() as source:
        try:
            audio = r.listen(source, timeout=15, phrase_time_limit=10)
            text = r.recognize_google(audio)
            logger.info(f"🎤 You said: {text}")
            return {
                "user_input": text,
                "messages": [{"role": "user", "content": text}]
            }
        except (sr.UnknownValueError, sr.RequestError) as e:
            logger.info("❌ Could not understand audio")
            return {
                "user_input": "",
                "messages": [{"role": "user", "content": ""}]
            }

graph_builder.add_node("listen", listen)

def chatbot(state: State):
    """Generate a response using LLM."""
    if not state["user_input"]:
        return {
            "chatbot_response": "I couldn't understand that. Could you please repeat?",
            "messages": [{"role": "assistant", "content": "I couldn't understand that. Could you please repeat?"}]
        }
    
    try:
        response = llm.invoke(state["messages"])
        logger.info(f"💡 Response generated")
        return {
            "chatbot_response": response,
            "messages": [{"role": "assistant", "content": response}]
        }
    except Exception as e:
        logger.info("❌ Error generating response")
        error_msg = "I'm sorry, I couldn't generate a response."
        return {
            "chatbot_response": error_msg,
            "messages": [{"role": "assistant", "content": error_msg}]
        }

graph_builder.add_node("chatbot", chatbot)

def speak(state: State):
    """Convert text response to speech using pyttsx3."""
    logger.info("🔊 Speaking response")
    try:
        engine.say(state['chatbot_response'])
        engine.runAndWait()
    except Exception:
        logger.info("❌ Error in text-to-speech")
    
    if state["user_input"].lower() == "exit":
        logger.info("👋 Ending conversation")
        return END
    return "listen"

graph_builder.add_node("speak", speak)

# Define the edges
graph_builder.add_edge(START, "listen")
graph_builder.add_edge("listen", "chatbot")
graph_builder.add_edge("chatbot", "speak")
graph_builder.add_edge("speak", "listen")

# Compile the graph
graph = graph_builder.compile()

# Initialize the state
initial_state = {
    "messages": [],
    "user_input": "",
    "chatbot_response": ""
}

# Run the graph
def main():
    logger.info("🎯 Starting conversation")
    print("Starting the conversation. Say 'exit' to quit.")
    try:
        for event in graph.stream(initial_state):
            pass
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
        print("\nGoodbye!")

if __name__ == "__main__":
    main()