from llama_cpp import Llama 
from argparse import ArgumentParser
import timeit

def main():
    parser = ArgumentParser()
    
    # Add the model path as a command-line argument
    parser.add_argument("-m", "--model_path", help="Path to the model",
                         dest="model_path", required=True)
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Load Llama 2 model
    llm = Llama(model_path=args.model_path,
                n_ctx=512,
                n_batch=128)

    # Start timer
    start = timeit.default_timer()

    # Generate LLM response
    prompt = "問題:誰是莎士比亞"

    output = llm(prompt,
                 max_tokens=-1,
                 echo=True,
                 top_p=0.9)

    # Stop timer
    stop = timeit.default_timer()
    duration = stop - start
    print("\nTime: ", duration, '\n\n')

    # Display generated text
    print(output['choices'][0]['text'])

    # Write to file
    write_txt(args.model_path, prompt, duration, output)

def write_txt(model_path, prompt, duration, output):
    with open("response.txt", "a", encoding="utf-8") as f:
        f.write(f"{model_path}\n")
        f.write(f"Prompt: {prompt}\n")  
        f.write(f"Time: {duration}\n")  
        f.write(f"Output: {output['choices'][0]['text']}\n\n")  

if __name__ == '__main__':
    main()
