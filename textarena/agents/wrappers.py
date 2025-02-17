import re 

from typing import Optional, Any, List, Tuple 
from e2b_code_interpreter import Sandbox, Logs

import textarena as ta 


__all__ = [ 
    "ThoughtAgentWrapper",
    "CodeInterpreterAgentWrapper",
    "ActorCriticAgentWrapper",
    "ChainAgentWrapper",
    "OpponentModelingAgentWrapper",
    "StrategicPerspectiveAgentWrapper",
    "RefinementAgentWrapper",
    "AnswerTokenAgentWrapper"
]

class AnswerTokenAgentWrapper(ta.AgentWrapper):
    """ TODO """
    def __init__(
        self,
        agent: ta.Agent,
        answer_token: Optional[str] = "### Answer:",
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



class StrategicPerspectiveAgentWrapper(ta.AgentWrapper):
    """
    This wrapper asks the agent to provide multiple perspectives on the current situation.
    Each perspective can yield a different proposed move. The wrapper then chooses or
    synthesizes a final move.
    """

    def __init__(
        self,
        agent: ta.Agent,
        perspectives: list = None,
        debugging: bool = False
    ):
        """
        :param agent: The underlying agent.
        :param perspectives: A list of perspective names (e.g. ['Offensive', 'Defensive']).
        :param debugging: Print debug information if True.
        """
        super().__init__(agent)
        self.perspectives = perspectives or ["Offensive", "Defensive", "Balanced"]
        self.debugging = debugging

    def evaluate_move(self, move: str) -> float:
        """
        A placeholder heuristic to evaluate how 'good' a move is.
        Override or improve as needed.
        """
        # Example: longer moves might be more elaborate, or use any domain-specific logic
        return len(move)

    def __call__(self, observation: str) -> str:
        original_system_prompt = self.agent.system_prompt

        moves = {}
        for perspective in self.perspectives:
            # Prompt the agent to analyze from a specific perspective
            self.agent.system_prompt = (
                f"You are analyzing the game from a {perspective} perspective. "
                "Provide your best move considering this perspective."
            )
            move = self.agent(observation)
            moves[perspective] = move
            if self.debugging:
                print(f"[{perspective} Perspective Move]\n{move}\n")

        # Decide which move is best
        best_perspective = None
        best_score = float("-inf")
        for perspective, move in moves.items():
            score = self.evaluate_move(move)
            if score > best_score:
                best_score = score
                best_perspective = perspective

        final_move = moves[best_perspective] if best_perspective else ""
        if self.debugging:
            print(f"[Chosen Perspective: {best_perspective} | Score: {best_score}]\n")
            print(f"[Final Move]\n{final_move}\n")

        # Restore the original system prompt
        self.agent.system_prompt = original_system_prompt

        return final_move



class ActorCriticAgentWrapper(ta.AgentWrapper):
    """ TODO """
    def __init__(
        self,
        agent: ta.Agent,
        actor_system_prompt: Optional[str] = None,
        critic_system_prompt: Optional[str] = None,
        final_reply_system_prompt: Optional[str] = None,
        debugging: bool = False
    ):
        """ TODO """
        super().__init__(agent)

        # Initialize prompts
        self.actor_system_prompt = (
            actor_system_prompt or 
                "You are an expert player in a competitive game environment. Your goal is to play optimally, "
                "adhering strictly to the game's rules and format requirements.\n\n"
                "Evaluate the current game state carefully and decide your next move strategically. "
                "Always ensure your response is:\n"
                "1. In the correct format as specified by the game rules.\n"
                "2. Focused on maximizing your chances of winning while minimizing risks.\n\n"
                "Respond only with your chosen move in the required format."
            )
        self.critic_system_prompt = (
            critic_system_prompt or
                "You are a critical observer and rule enforcer for a competitive game. Your task is to:\n"
                "1. Review the provided action to ensure it complies with the game's rules and format requirements.\n"
                "2. Critique the action for strategic effectiveness, identifying any potential improvements. "
                "[esp. pointing out if this action will give the opponent a chance to win in the next move]\n"
                "3. Highlight specific reasons if the action is invalid or suboptimal.\n\n"
                "Provide your analysis clearly and concisely. Never trading is not a good strategy! Accept trades that seem good!"
            )
        self.final_reply_system_prompt = (
            final_reply_system_prompt or
                "You are an expert game strategist tasked with synthesizing a final move. Based on the initial action and the critique, "
                "generate a revised action that:\n"
                "1. Adheres strictly to the game's rules and format.\n"
                "2. Incorporates the improvements or corrections suggested in the critique.\n"
                "3. Optimizes strategic effectiveness to maximize the chances of success.\n\n"
                "Respond only with the final corrected action in the required format. Try not to make new offers every single move."
            )

        self.debugging = debugging

    def __call__(self, observation: str) -> str:
        """ TODO """

        # set first system prompt 
        self.agent.system_prompt = self.actor_system_prompt

        # first forward 
        initial_out = self.agent(observation)
        if self.debugging:
            print(f"Initial Agent output: {initial_out}")

        # set second system prompt 
        self.agent.system_prompt = self.critic_system_prompt

        # second forward
        second_out = self.agent(f"Observation: {observation}. \nCurrent Strategy: {initial_out}")
        if self.debugging:
            print(f"Critic Agent output: {second_out}")

        # set third system prompt 
        self.agent.system_prompt = self.final_reply_system_prompt

        # third forward
        final_out = self.agent(
            f"Observation: {observation}. \nInitial Strategy: {initial_out}. \nCriticism: {second_out} "
        )
        if self.debugging:
            print(f"Final Agent output: {final_out}\n\n")


        return final_out 
           

class RefinementAgentWrapper(ta.AgentWrapper):
    """
    This wrapper calls the underlying agent multiple times (n refinement steps).
    Each step sees:
      - The original observation
      - The previously drafted actions
    The agent is asked to refine or improve upon the last draft. 
    The final draft is returned after n steps.
    """
    def __init__(
        self,
        agent: ta.Agent,
        n_refinements: int = 3,
        iteration_prompt: Optional[str] = None,
        debugging: bool = False
    ):
        """
        :param agent: The underlying agent that will be called repeatedly.
        :param n_refinements: Number of refinement steps.
        :param iteration_prompt: A prompt template for each refinement iteration.
            It should expect two placeholders:
                {iteration} (the current iteration index)
                {previous_actions} (the concatenated previous drafts)
            If not provided, a default prompt is used.
        :param debugging: If True, prints debug info for each refinement step.
        """
        super().__init__(agent)
        self.n_refinements = n_refinements
        self.iteration_prompt = iteration_prompt or (
            "You are refining your answer for iteration {iteration}. "
            "Below are your previous drafts:\n{previous_actions}\n\n"
            "Please provide an improved or corrected final action in valid format."
        )
        self.debugging = debugging

    def __call__(self, observation: str) -> str:
        """
        Executes n_refinements steps. Each step sees observation + all previous drafts.
        Returns the last refinement's output.
        """
        original_system_prompt = self.agent.system_prompt
        drafts = []  # store the agent's outputs from each refinement step

        try:
            for i in range(self.n_refinements):
                # Prepare the iteration prompt
                iteration_text = self.iteration_prompt.format(
                    iteration=i + 1,
                    previous_actions="\n".join(
                        f"Draft {idx+1}: {draft}" for idx, draft in enumerate(drafts)
                    )
                )

                # Update system prompt for the iteration
                self.agent.system_prompt = iteration_text

                # Construct combined input: observation + the prompt
                combined_input = observation
                if self.debugging:
                    print(f"\n=== Refinement Iteration {i+1} ===")
                    print(f"System Prompt:\n{self.agent.system_prompt}")
                    print(f"Agent Input:\n{combined_input}")

                # Get the draft
                draft = self.agent(combined_input).strip()
                drafts.append(draft)

                if self.debugging:
                    print(f"Draft {i+1} Output:\n{draft}")

            # The final draft after n refinements
            final_output = drafts[-1] if drafts else ""
            return final_output

        finally:
            # Restore original system prompt
            self.agent.system_prompt = original_system_prompt



class CodeInterpreterAgentWrapper(ta.AgentWrapper):
    """
    An improved code-interpreter-based wrapper. It guides the agent to:
      1) Plan out how to compute or analyze the game situation.
      2) Write code to do the computations.
      3) Execute the code in a sandbox, collect logs (stdout/stderr).
      4) Summarize or interpret the results, and produce a final answer.

    If code execution fails or there are errors, it gives the agent a chance
    to correct the code.
    """

    def __init__(
        self,
        agent: ta.Agent,
        plan_prompt: Optional[str] = None,
        code_prompt: Optional[str] = None,
        fix_prompt: Optional[str] = None,
        summarize_prompt: Optional[str] = None,
        prompt_max_chars: int = 8000,
        logs_max_chars: int = 4000,
        debugging: bool = False
    ):
        super().__init__(agent)
        self.sbx = Sandbox()

        # Prompts
        self.plan_prompt = plan_prompt or (
            "First, outline a plan for any necessary computations or analysis. "
            "If code is needed, you will provide it in the next step."
        )
        self.code_prompt = code_prompt or (
            "Now provide Python code (in a single code block ```python ... ``` ) that implements your plan. "
            "Do not repeat large amounts of previously-written code. Use short functions and references."
        )
        self.fix_prompt = fix_prompt or (
            "The code encountered an error. Please revise it to fix the issue. "
            "Provide a corrected code block."
        )
        self.summarize_prompt = summarize_prompt or (
            "Given the execution logs above, summarize the results of your computations "
            "and provide your final game move or answer in the correct format."
        )

        self.prompt_max_chars = prompt_max_chars
        self.logs_max_chars = logs_max_chars
        self.debugging = debugging

        # Stores (code_block, logs) for all executions
        self.executed_code: List[Tuple[str, Logs]] = []

    def _extract_code_blocks(self, text: str) -> List[str]:
        return re.findall(r"```python\s*([\s\S]*?)```", text, flags=re.DOTALL)

    def _format_logs_for_prompt(self, logs: Logs) -> str:
        stdout = "".join(logs.stdout).strip()
        stderr = "".join(logs.stderr).strip()

        output = ""
        if stdout:
            output += f"<stdout>\n{stdout}\n</stdout>\n"
        if stderr:
            output += f"<stderr>\n{stderr}\n</stderr>\n"
        return output

    def _merged_logs_text(self) -> str:
        """
        Merges all execution logs (stdout, stderr) into a single string,
        truncated to logs_max_chars if necessary.
        """
        all_logs = []
        for code_block, logs in self.executed_code:
            section = f"Code:\n```python\n{code_block}\n```\n{self._format_logs_for_prompt(logs)}"
            all_logs.append(section)
        merged = "\n\n".join(all_logs)
        return merged[-self.logs_max_chars:]

    def _try_execute_code(self, code_block: str) -> Logs:
        try:
            execution = self.sbx.run_code(code_block, timeout=5)
            return execution.logs
        except Exception as e:
            error_log = Logs(stdout=[], stderr=[f"Code execution error: {str(e)}"])
            return error_log

    def __call__(self, observation: str) -> str:
        """
        1) Plan pass
        2) Code generation pass
        3) Code execution (with optional error fixes)
        4) Summarize final move
        """
        original_system_prompt = self.agent.system_prompt
        
        try:
            # 1. Planning phase
            self.agent.system_prompt = f"{original_system_prompt}\n\n{self.plan_prompt}"
            plan_response = self.agent(observation)
            
            if self.debugging:
                print(f"Plan:\n{plan_response}\n")

            # 2. Code generation phase
            self.agent.system_prompt = f"{original_system_prompt}\n\n{self.code_prompt}"
            code_response = self.agent(f"{observation}\n\nPlan:\n{plan_response}")
            code_blocks = self._extract_code_blocks(code_response)
            
            if not code_blocks:
                return plan_response  # No code needed, return the plan analysis
            
            # 3. Code execution phase with error handling
            max_attempts = 3
            current_attempt = 0
            final_logs = None
            
            while current_attempt < max_attempts:
                code_block = code_blocks[-1]  # Use the last code block
                logs = self._try_execute_code(code_block)
                self.executed_code.append((code_block, logs))
                
                if self.debugging:
                    print(f"Execution {current_attempt + 1}:")
                    print(self._format_logs_for_prompt(logs))
                
                # Check for errors in stderr
                if not any(logs.stderr):
                    final_logs = logs
                    break
                
                # If there are errors, try to fix the code
                self.agent.system_prompt = f"{original_system_prompt}\n\n{self.fix_prompt}"
                fix_context = (
                    f"{observation}\n\n"
                    f"Previous code:\n```python\n{code_block}\n```\n\n"
                    f"Error logs:\n{self._format_logs_for_prompt(logs)}"
                )
                
                fix_response = self.agent(fix_context)
                code_blocks = self._extract_code_blocks(fix_response)
                
                if not code_blocks:
                    break  # No more code to try, exit the loop
                
                current_attempt += 1
            
            # 4. Summarization phase
            self.agent.system_prompt = f"{original_system_prompt}\n\n{self.summarize_prompt}"
            
            # Prepare context with all execution history
            execution_history = self._merged_logs_text()
            summary_context = (
                f"{observation}\n\n"
                f"Original plan:\n{plan_response}\n\n"
                f"Execution history:\n{execution_history}"
            )
            
            # Truncate if necessary
            if len(summary_context) > self.prompt_max_chars:
                summary_context = summary_context[-self.prompt_max_chars:]
            
            final_response = self.agent(summary_context)
            return final_response

        finally:
            # Restore original system prompt
            self.agent.system_prompt = original_system_prompt



class ChainAgentWrapper(ta.Agent):
    """
    Chains several agents and pipes the output of each agent to the next
    """
    def __init__(self, agents: List[ta.Agent], delimiter: str = "\n"):
        super().__init__()
        self.agents = agents
        self.delimiter = delimiter

    def __call__(self, observation: str) -> str:
        current_observation: str = f"Observation: {observation}\n"
        for i, agent in enumerate(self.agents):
            if i == len(self.agents)-1:
                out_text = agent(current_observation+"\nFinal Output: ")
            else:
                out_text = agent(current_observation)
                current_observation += f"Step {i+1} Output:{out_text}\n"
        return out_text
