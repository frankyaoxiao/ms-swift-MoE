#!/usr/bin/env python3
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")

system_prompt = ""
messages = []

def reset():
    global messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    print("[Context cleared]")

def set_system(prompt):
    global system_prompt
    system_prompt = prompt
    reset()
    print(f"[System prompt set to: {prompt[:50]}{'...' if len(prompt) > 50 else ''}]")

def chat(user_input):
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="default",
        messages=messages,
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "separate_reasoning": True,
        }
    )
    thinking = getattr(response.choices[0].message, 'reasoning_content', None)
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return thinking, reply

print("Commands:")
print("  /system <prompt>  - Set system prompt (also clears context)")
print("  /clear            - Clear conversation context")
print("  /quit             - Exit")
print()

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
        break

    if not user_input:
        continue

    if user_input.lower() == "/quit":
        break
    elif user_input.lower() == "/clear":
        reset()
    elif user_input.lower().startswith("/system "):
        set_system(user_input[8:])
    elif user_input.startswith("/"):
        print(f"[Unknown command: {user_input.split()[0]}]")
    else:
        try:
            thinking, reply = chat(user_input)
            if thinking:
                print(f"\n<think>\n{thinking}\n</think>\n")
            print(f"Assistant: {reply}\n")
        except Exception as e:
            print(f"[Error: {e}]")
