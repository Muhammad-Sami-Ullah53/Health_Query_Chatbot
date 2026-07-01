"""
Task 4: General Health Query Chatbot (Prompt Engineering)

Goal:
Create a chatbot that can answer general health-related questions
using a Large Language Model (LLM).

Objective:
Use Qwen2.5-0.5B-Instruct from Hugging Face to build a simple
health chatbot without requiring any paid API.

Requirements:
    pip install -U transformers torch accelerate sentencepiece safetensors
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ============================================================
# Load Model
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

# ============================================================
# Device Configuration
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Running on: {device}")

# ============================================================
# Conversation Memory
# ============================================================

conversation = [
    {
        "role": "system",
        "content":
        (
            "You are HealthBot, a professional and friendly AI medical assistant. "
            "Answer only general health-related questions. "
            "Keep answers short, accurate and easy to understand. "
            "Do not diagnose diseases. "
            "If symptoms are severe or persistent, recommend consulting a healthcare professional."
        )
    }
]

# ============================================================
# Chatbot Function
# ============================================================

def chatbot_response(user_input):

    conversation.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    conversation.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    return response


# ============================================================
# Main Chat Loop
# ============================================================

print("=" * 60)
print(" General Health Query Chatbot")
print("=" * 60)
print("Ask your health-related questions.")
print("Type 'exit' or 'quit' to end.")
print("=" * 60)

while True:

    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
        print("\nBot: Goodbye! Stay healthy.")
        break

    print("\nBot is typing...\n")

    reply = chatbot_response(user_input)

    print("Bot:", reply)

# ============================================================
# Notes
# ============================================================

"""
Limitations
-----------
• This chatbot provides only general health information.
• It is not a substitute for professional medical advice.
• For emergencies or serious symptoms, consult a qualified doctor.

Example Questions
-----------------
• What causes a sore throat?
• What are the symptoms of fever?
• What is influenza?
• How can I avoid getting sick?
• What should I do for a headache?
• What foods improve immunity?
• How much water should I drink daily?
"""