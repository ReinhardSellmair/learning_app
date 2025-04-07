import re

def remove_angle_brackets(text: str) -> str:
    return re.sub(r'<.*?>', '', text)
