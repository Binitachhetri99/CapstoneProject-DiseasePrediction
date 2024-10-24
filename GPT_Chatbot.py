import tkinter as tk
from tkinter import scrolledtext
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
import threading
import time  # Import the time module for timing

# Global variables to hold the model and processor
model = None
processor = None

def initialize_model():
    global model, processor
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    print("Initializing model...")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model initialized.")

def llama_chatbot_response(user_input):
    # Prepare the message for the model
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": user_input},
            {"type": "text", "text": "Please provide a response."}
        ]}
    ]
    
    # Process the message to apply chat template
    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Prepare inputs for the model
    inputs = processor(
        text=text,
        return_tensors="pt"
    ).to(model.device)

    start_time = time.time()  # Start timing before the model generates a response

    output = model.generate(**inputs, max_new_tokens=5)
    response = processor.decode(output[0], skip_special_tokens=True)

    end_time = time.time()  # End timing after the response is generated

    # Trim the response to remove any unwanted system information
    response = response.split("user")[1].strip() if "user" in response else response.strip()

    # Calculate the time taken in seconds
    duration = end_time - start_time

    print(f"Bot response: {response}, Time taken: {duration:.2f} seconds")  # Debug print

    # Return both the response and the time taken for display in the UI
    return response, duration

class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Chatbot")
        
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', width=60, height=20)
        self.chat_area.grid(column=0, row=0, padx=10, pady=10)
        
        self.text_input = tk.Entry(root, width=60)
        self.text_input.grid(column=0, row=1, padx=10, pady=10)

        self.send_button = tk.Button(root, text="Send", command=self.start_sending_message)
        self.send_button.grid(column=0, row=2, padx=10, pady=10)
        
        self.text_input.bind("<Return>", self.start_sending_message)

    def start_sending_message(self, event=None):
        user_message = self.text_input.get()
        if user_message:
            # Clear the input field immediately
            self.text_input.delete(0, tk.END)

            # Display the user's message
            self.chat_area.configure(state='normal')
            self.chat_area.insert(tk.END, "You: " + user_message + "\n")
            self.chat_area.insert(tk.END, "Bot: Thinking...\n")  # Indicate that the bot is processing
            self.chat_area.configure(state='disabled')

            threading.Thread(target=self.send_message, args=(user_message,)).start()

    def send_message(self, user_message):
        # Get bot's response and the time taken
        try:
            bot_response, response_time = llama_chatbot_response(user_message)
        except Exception as e:
            bot_response, response_time = f"Error: {str(e)}", 0.0
        
        # Display bot's response and the time taken
        self.chat_area.configure(state='normal')

        # Clear previous "Thinking..." message and insert the new response
        self.chat_area.delete("end-2l", tk.END)  # Remove the "Bot: Thinking..." line
        self.chat_area.insert(tk.END, f"Bot: {bot_response} (Response time: {response_time:.2f} seconds)\n")
        
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)

if __name__ == "__main__":
    initialize_model()  # Initialize the model and processor
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
