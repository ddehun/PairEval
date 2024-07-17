TARGET_TOKENS = {"pairwise": ["A", "B"]}

PROMPT_DICT = {
    "turn_level": {
        "pairwise": (
            "[INST] Instruction: Given two conversations and responses, which response is better for the conversation?\n\n"
            "{knowledge_1}Conversation A:\n{history_1}\n\nResponse A:\n{response_1}\n\n"
            "{knowledge_2}Conversation B:\n{history_2}\n\nResponse B:\n{response_2}\n\n"
            "Which response is better for the conversation? "
            'Just answer in "Response A" or "Response B". [/INST]\n'
            "Answer: Response"
        ),
    }
}
