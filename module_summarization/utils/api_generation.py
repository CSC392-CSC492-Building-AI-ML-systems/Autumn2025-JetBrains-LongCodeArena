def gpt_generation(client, prompt, model_name='gpt-3.5-turbo-16k', temp=0.0):
    if "gpt-5" in model_name.lower():
        full_prompt = "You are a helpful assistant.\n\nUser request:\n" + prompt
        response = client.responses.create(
            model=model_name,
            input=full_prompt,
            text={"verbosity": "low"},
        )
        return response.output_text
    
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            n=1,
            stream=False,
            temperature=temp,
            max_tokens=2000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content