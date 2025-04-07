from config import N_KEYWORDS, N_CHOICES, FAILED_QUERY

# template for keyword prompt
KEYWORDS_TEMPLATE = """
You're an assistant to generate keywords to search for wikipedia articles that contain content the user wants to learn. 
For a given user query return at most {n_keywords} keywords. Make sure every keyword is a good match to the user query. 
Rather provide fewer keywords than keywords that are less relevant.

Instructions:
- Return the keywords separated by commas 
- Do not return anything else
"""

# template for question generation prompt
MCQ_TEMPLATE = """
You are a learning app that generates multiple-choice questions based on educational content. The user provided the 
following request to define the learning content:

"{user_query}"

Based on the user request, following context was retrieved:

"{context}"

Generate a multiple-choice question directly based on the provided context. The correct answer must be explicitly stated 
in the context and should always be the first option in the choices list. Additionally, provide an explanation for why 
the correct answer is correct.
Number of answer choices: {n_choices}
{previous_questions}{rejected_questions}
The JSON output should follow this structure (for number of choices = 4):

{{"question": "Your generated question based on the context", "choices": ["Correct answer (this must be the first choice)","Distractor 1","Distractor 2","Distractor 3"], "explanation": "A brief explanation of why the correct answer is correct."}}

Instructions:
- Generate one multiple-choice question strictly based on the context.
- Provide exactly {n_choices} answer choices, ensuring the first one is the correct answer.
- Include a concise explanation of why the correct answer is correct.
- Do not return anything else than the json output.
- The provided explantion should not assume the user is aware of the context. Avoid formulations like "As stated in the text...".
- The response must be machine reaadable and not contain line breaks.
- Check if it is possible to generate a question based on the provided context that is aligned with the user reqeust. If it is not possible set the generated question to "{fail_keyword}".
"""

def keyword_sys_prompt(n=N_KEYWORDS):
    return KEYWORDS_TEMPLATE.format(n_keywords=str(int(n)))

def question_generation_prompt(user_query, context, questions_df, n_choices=N_CHOICES):
    previous_questions = ""
    rejected_questions = ""

    # check if there are any questions
    if not questions_df.empty:
        # get previous questions
        previous_questions = "\nPreviously following questions have been generated:\n"
        previous_questions += '\n'.join(questions_df.apply(lambda x: f'- question: "{x["question"]}", answer: {x["correct_answer"]}\n', axis=1))
        for _, row in questions_df.iterrows():
            previous_questions += f'- question: "{row["question"]}", answer: {row["correct_answer"]}\n'
        previous_questions += "Do not repeat any of the previous questions in the new question generation.\n"

        # check if there are rejected questions
        if questions_df['question_rejected'].any():
            # get rejected questions
            rejected_questions = "\nThe following questions have been rejected by the user:\n"
            for _, row in questions_df.loc[questions_df['question_rejected']].iterrows():
                rejected_questions += f'- question: "{row["question"]}", answer: {row["correct_answer"]}\n'
            rejected_questions += "Avoid generating questions of a similar nature and context.\n"

    # fill prompt template
    prompt = MCQ_TEMPLATE.format(user_query=user_query, context=context, previous_questions=previous_questions, 
                                 rejected_questions=rejected_questions, n_choices=n_choices, fail_keyword=FAILED_QUERY)

    return prompt
