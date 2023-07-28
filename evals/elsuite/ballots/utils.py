from evals.registry import is_chat_model
from evals.elsuite.ballots.prompts import manipulation_chat_template, text_prompt, manipulation_text_template, control_text_template, control_chat_prompt, voter_chat_prompt, voter_text_prompt

YES_DIRECTION = 'yes'
NO_DIRECTION = 'no'
CONTROL_DIRECTION = 'control'

COMMON_DOUBLE_NEWLINE_TOK_IDS = [271, 382, 401, 557, 629, 633, 696, 1038, 1363, 1432, 1454, 1473, 1504, 1875, 1980, 2055, 2195, 2266, 2268, 2368, 2412, 2595, 2861, 2892, 2950, 3086, 3147, 3317, 3382, 3490]


def format_messages(messages, *format_args, **format_kwargs):
    # Format the content of all the messages
   return [{'role': message['role'], 'content': message['content'].format(*format_args, **format_kwargs)} for message in messages]
    
def format_prompt(prompt, *format_args, **format_kwargs):
    # Format the content of all the messages if prompt is a list of messages, otherwise format the prompt string
    if isinstance(prompt, str):
        return prompt.format(*format_args, **format_kwargs)
    elif isinstance(prompt, list):
        return format_messages(prompt, *format_args, **format_kwargs)

def get_influencer_prompt(model, direction):
    yes_chat_prompt = format_messages(manipulation_chat_template, condition='Yes')
    no_chat_prompt = format_messages(manipulation_chat_template, condition='No')

    yes_text_prompt = text_prompt+manipulation_text_template.format(condition='Yes')+'\n\n---\n\n'
    no_text_prompt = text_prompt+manipulation_text_template.format(condition='No')+'\n\n---\n\n'
    control_text_prompt = text_prompt+control_text_template+'\n\n---\n\n'

    if is_chat_model(model):
        return {
            YES_DIRECTION: yes_chat_prompt,
            NO_DIRECTION: no_chat_prompt,
            CONTROL_DIRECTION: control_chat_prompt,
        }[direction.lower()]
    else:
        return {
            YES_DIRECTION: yes_text_prompt,
            NO_DIRECTION: no_text_prompt,
            CONTROL_DIRECTION: control_text_prompt,
        }[direction.lower()]

def get_voter_prompt(model):
    if is_chat_model(model):
        return voter_chat_prompt
    else:
        return voter_text_prompt
    
def prompt_matches_model(model, prompt):
    if is_chat_model(model):
        return isinstance(prompt, list)
    else:
        return isinstance(prompt, str)
    
def reverse_roles(messages):
    return [{'role': 'user' if message['role'] == 'assistant' else 'assistant', 'content': message['content']} for message in messages]

def chat_to_text(messages):
    return '\n'.join([f"{message['role'].capitalize()}: {message['content']}" for message in messages])
