import os
import inspect

from together import Together

from .example_generation_model import ExampleGenerationModel


class TogetherModel(ExampleGenerationModel):

    def __init__(self, model_name: str, use_bm25: bool = False, n_selections: int = 0):
        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.model_name = model_name
        self.use_bm25 = use_bm25
        self.n_selections = n_selections

    def generate(self, task_description: str, project_apis: list[str] = None) -> str:
                
        if not self.use_bm25:
            instruction = self.get_prompt(task_description)
        
            prompt = [
                {"role": "user", "content": instruction}
            ]
            
        else:
            instruction, recommended = self.get_bm25_prompt(task_description, project_apis, n_selections=self.n_selections)
            extra_context = [f"{elem}: {str(inspect.signature(elem) if callable(elem) else 'None') }\n" for elem in recommended]
            
            prompt = [
                {"role": "user", "content": instruction + "\nHere is a list of function headers for each project element:\n" + '\n'.join(extra_context)},
            ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            temperature=0.0,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content

    def name(self):
        if not self.use_bm25:
            return self.model_name
        else:
            return f"bm25/{self.model_name}"
