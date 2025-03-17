
def load_prompt(filename: str, language: str) -> str:
    """Helper to load prompt text from file"""
    prompt_folder = "prompts/chinese_prompts" if language == "chinese" else "prompts"
    with open(f"textarena/envs/Diplomacy/{prompt_folder}/{filename}", "r") as f:
        return f.read().strip()

def get_state_specific_prompt(state: str, language: str) -> str:
    return load_prompt(f"state_specific/{state.lower()}_system_prompt.txt", language)
