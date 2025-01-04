def get_valid_question_prompt(user_input: str):
    return f"""
                Identify the following items from the Input Text: 
                - is there any valid question present in the input text ? (True or False)

                Input Text is delimited with <>. \
                Format your response as a JSON object with \
                "valid_question" as the keys.
                If the information isn't present or the question is not complete in itself, use "None" as the value. Do not make it up.
                Make your response as short as possible.
                Format the valid_question value as a boolean.
                Input Text: <{user_input}>
                """

GET_ANSWER_PROMPT = """
You are an expert Teacher, who loves to teach/answer questions.
Always try to answer the question in the context of any object identified in the image. Provide a brief information about the identified object.
Don't mention the word image in the response. Response should be in 1st person tone.

Example.
It looks like you are holding a samsung mobile phone ...

Your answer should not exceed more than 100 words.
"""
