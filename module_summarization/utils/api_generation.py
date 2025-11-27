def gpt_generation(client, prompt, model_name="gpt-5-nano"):
    full_prompt ="You are a helpful assistant.\n\nUser request:\n" + prompt


    response = client.responses.create(
        model=model_name,
        input=full_prompt,
        text={ "verbosity": "low" },
    )

    return response.output_text
