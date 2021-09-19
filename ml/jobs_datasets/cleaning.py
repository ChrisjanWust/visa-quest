def is_valid(key, text):
    if not isinstance(text, str):
        return False
    if "<div" in text:
        return False
    if key == "salary":
        if not any(char.isdigit() for char in text):
            return False
    return True
