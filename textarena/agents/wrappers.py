import re 
from typing import Optional, Any, List, Tuple 

import textarena as ta 

__all__ = ["AnswerTokenAgentWrapper", "ThoughtAgentWrapper"]


class AnswerTokenAgentWrapper(ta.AgentWrapper):
    """ TODO """
    def __init__(
        self,
        agent: ta.Agent,
        answer_token: Optional[str] = "Final Answer",
        debugging: bool = False
    ):
        """ TODO """
        super().__init__(agent)
        self.answer_token = answer_token
        self.debugging = debugging


    def __call__(self, observation: str) -> str:
        """ TODO """

        # set the agent prompt just for this part
        current_system_prompt = self.agent.system_prompt 
        answer_token_prompt = current_system_prompt + \
            f"Anything you return after '{self.answer_token}' will be submitted to the game."

        self.agent.system_prompt = answer_token_prompt
        if self.debugging:
            print(f"Model System prompt: {answer_token_prompt}")
        
        raw_answer = self.agent(observation)

        # reset prompt 
        self.agent.system_prompt = current_system_prompt

        if self.debugging:
            print(f"Model raw output: {raw_answer}")
        if self.answer_token in raw_answer:
            if self.debugging:
                print(f"Model filtered output: {raw_answer.split(self.answer_token)[-1]}")
            return raw_answer.split(self.answer_token)[-1]

        else:
            return raw_answer



class ThoughtAgentWrapper(ta.AgentWrapper):
    """ TODO """
    def __init__(
        self, 
        agent:ta.Agent, 
        thought_prompt: Optional[str] = None, 
        answer_prompt: Optional[str] = None,
        debugging: bool = False
    ):
        """ TODO """
        super().__init__(agent)

        self.agent_system_prompt = self.agent.system_prompt
        self.thought_prompt = thought_prompt if thought_prompt is not None else (
            "\nPlease think extensively about what you want to do next. Analyze your current position, "
            "you strategy, what your opponents strategy might be and what you should do next to maximize "
            "your chance of winning."
        )

        self.answer_prompt = answer_prompt if answer_prompt is not None else (
            "\nGiven the game observations, and your above thoughts, please give the reply you want "
            "to submit to the game. Make sure you follow all rules and necessary formats."
        )

        self.debugging = debugging 


    def __call__(self, observation: str) -> str:
        """ TODO """

        # set agent prompt 
        self.agent.system_prompt = self.thought_prompt

        # first forward
        thoughts = self.agent(observation + f"\n\nThoughts: ")
        if self.debugging:
            print(f"\n\nAgent thoughts: {thoughts}")


        # set agent prompt 
        self.agent.system_prompt = self.answer_prompt 

        # second forward
        answer = self.agent(observation + f"\n\nThoughts: {thoughts}" + self.answer_prompt)

        if self.debugging:
            print(f"\n\nAnswer: {answer}")
        return answer 

