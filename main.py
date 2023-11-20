from llama_cpp import Llama 
import timeit

# Load Llama 2 model
llm = Llama(model_path="./model/llama-2-7b-chat.Q2_K.gguf",
            n_ctx=512,
            n_batch=128)

# Start timer
start = timeit.default_timer()

# Generate LLM response
prompt = "Who wrote Romeo and Juliet?"

output = llm(prompt,
             max_tokens=100,
             echo=False,
             temperature=0.1,
             top_p=0.9)

# Stop timer
stop = timeit.default_timer()
duration = stop - start
print("Time: ", duration, '\n\n')

# Display generated text
print(output['choices'][0]['text'])

# Write to file
with open("response.txt", "w") as f:
  f.write(f"Time: {duration}")
  f.write(output['choices'][0]['text'])