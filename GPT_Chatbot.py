import tkinter as tk
from tkinter import filedialog, scrolledtext, Entry, Button, END
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
import threading
import time
from PIL import Image
import re

class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multimodal Chatbot")

        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', width=60, height=20)
        self.chat_area.grid(column=0, row=0, columnspan=2, padx=10, pady=10)

        self.text_input = Entry(root, width=60)
        self.text_input.grid(column=0, row=1, padx=10, pady=10)

        # Bind the return key to send the message
        self.text_input.bind("<Return>", self.start_sending_text)

        self.send_button = Button(root, text="Send Text", command=self.start_sending_text)
        self.send_button.grid(column=0, row=2, padx=10, pady=10)

        self.photo_button = Button(root, text="Send Photo", command=self.select_photo)
        self.photo_button.grid(column=1, row=2, padx=10, pady=10)

        # Initialize the model and processor
        self.model = None
        self.processor = None
        self.initialize_model()

    def initialize_model(self):
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        print("Initializing model...")

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        print("Model initialized.")

    def llama_chatbot_response(self, user_input=None, image=None):
        # Prepare the message content
        content = []
        if user_input:
            content.append({"type": "text", "text": user_input})
        if image:
            content.append({"type": "image", "image": image})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        if image:
            model_inputs = self.processor(
                text=text,
                images=[image],
                return_tensors="pt"
            ).to(self.model.device)
        else:
            model_inputs = self.processor(
                text=text,
                return_tensors="pt"
            ).to(self.model.device)

        start_time = time.time()
        output = self.model.generate(**model_inputs, max_new_tokens=9999)
        response = self.processor.decode(output[0], skip_special_tokens=True)
        duration = time.time() - start_time

        # Clean the response for display
        response = self.clean_response(response)

        return response, duration

    def clean_response(self, response):
        # Remove markdown symbols like ** and | and clean up sections
        response = re.sub(r"\*\*|(\|.*?\|)", "", response)  # Remove stars and pipes in tables
        response = re.sub(r"(Image Summary|Section|Content):", "", response)  # Remove these headers
        response = re.sub(r"\s{2,}", " ", response).strip()  # Remove extra whitespace

        # Further cleanup to remove any other remnants of unwanted formatting
        if "assistant" in response:
            response = response.split("assistant", 1)[-1].strip()
        
        return response

    def start_sending_text(self, event=None):
        user_message = self.text_input.get()
        if user_message:
            self.text_input.delete(0, END)
            self.display_message(f"You: {user_message}\nBot: Thinking...\n")
            threading.Thread(target=self.send_message, args=(user_message, None)).start()

    def select_photo(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            image = Image.open(file_path)
            self.display_message("You: [Image Selected]\nBot: Processing...\n")
            threading.Thread(target=self.send_message, args=(None, image)).start()

    def send_message(self, user_message=None, image=None):
        try:
            bot_response, response_time = self.llama_chatbot_response(user_message, image)
        except Exception as e:
            bot_response, response_time = f"Error: {str(e)}", 0.0

        self.display_message(f"Bot: {bot_response} (Response time: {response_time:.2f} seconds)\n")

    def display_message(self, message):
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, message)
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
