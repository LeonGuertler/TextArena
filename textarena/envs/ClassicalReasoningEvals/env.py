import re, random
import importlib.resources
from typing import Any, Dict, List, Tuple, Optional, Callable, Union
import multiprocessing
from math import isclose
from collections import defaultdict, Counter

import textarena as ta

try:
    import sympy
    from sympy import simplify, N
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.parsing.latex import parse_latex
    import latex2sympy2
    from latex2sympy2 import latex2sympy
except ImportError:
    raise ImportError(
        "sympy and latex2sympy2 required for the ClassicalReasoningEvals environment. "
        "Install it with: pip install sympy latex2sympy2"
    )


class ClassicalReasoningEvalsEnv(ta.Env):
    """Environment for Classical Reasoning Evaluations with multiple evaluation metrics."""
    
    VALID_EVAL_METHODS = ["accuracy", "x@k"]
    
    def __init__(self, file_name: str, n_samples: Optional[int] = None, eval_method: str="accuracy", k: int=16):
        self.file_name = file_name 
        self.n_sample = n_samples
        
        # Parse and validate eval_method
        if eval_method in ["pass@1", "pass@k", "cons@k", "best@k"]:
            # For backward compatibility, treat all *@k methods as "x@k"
            self.eval_method = "x@k"
            self.k = k
        elif '@' in eval_method:
            method_name, k_value = eval_method.split('@')
            try:
                self.k = int(k_value)
                self.eval_method = "x@k"
            except ValueError:
                self.eval_method = eval_method  # Keep original for error message
        else:
            self.eval_method = eval_method
            self.k = k
            
        if self.eval_method not in self.VALID_EVAL_METHODS:
            raise ValueError(f"Invalid eval_method: {eval_method}. Must be one of {self.VALID_EVAL_METHODS} or a legacy *@k format")

        # Load the dataset
        self._load_dataset()

    def _load_dataset(self):
        """
        Load dataset from the specified file in the textarena package's data directory.
        If n_sample is specified, randomly select that many samples from the dataset.
        """
        try:
            # Split the file path correctly
            path_parts = self.file_name.split('/')
            dataset_dir = path_parts[0]
            file_name = '/'.join(path_parts[1:])
            
            # Get the base package path
            package_path = "textarena.envs.ClassicalReasoningEvals.data"
            
            # Use importlib.resources to access the file
            data_pkg = importlib.import_module(f"{package_path}.{dataset_dir}")
            resource_path = importlib.resources.files(data_pkg) / file_name
            
            with open(resource_path, 'r') as f:
                # Parse the file based on extension
                if file_name.endswith('.jsonl'):
                    import json
                    self.dataset = [json.loads(line) for line in f if line.strip()]
                elif file_name.endswith('.json'):
                    import json
                    self.dataset = json.load(f)
                else:
                    # Default case: read as text
                    self.dataset = f.readlines()
            
            # If n_sample is specified, randomly select that many samples
            if self.n_sample is not None and self.n_sample < len(self.dataset):
                self.dataset = random.sample(self.dataset, self.n_sample)
            
            # turn it into an iterator
            self.total_questions = len(self.dataset)
            self.questions_seen = 0
            self.dataset = iter(self.dataset)
            self.question = None

        except Exception as e:
            raise ValueError(f"Failed to load dataset from {self.file_name}: {e}")

    def reset(self, num_players: int, seed: Optional[int]=None):
        """Reset the environment state for a new session."""
        self.state = ta.State(num_players=num_players, min_players=1, max_players=1)
        
        # Initialize game state with metrics for different evaluation methods
        game_state = {
            "questions_total": 0,
            "questions_correct": 0,  # For accuracy
            "current_question": {
                "attempts": [],      # List of booleans indicating correctness of each attempt
                "answers": []        # Store actual answers for consensus calculation
            },
            # For x@k method, track all metrics
            "x@k_metrics": {
                "pass@k": 0,
                "cons@k": 0,
                "best@k": 0,
                "total": 0
            }
        }
        
        self.state.reset(seed=seed, game_state=game_state)

        # Set the first question
        self._show_next_question()

    def _calculate_accuracy(self):
        """Calculate simple accuracy across all questions."""
        if self.state.game_state["questions_total"] == 0:
            return 0.0
        return self.state.game_state["questions_correct"] / self.state.game_state["questions_total"]
        
    def _calculate_pass_at_k(self):
        """Calculate pass@k metric for the current question."""
        attempts = self.state.game_state["current_question"]["attempts"]
        if not attempts:
            return 0.0

        # return average
        return sum([int(a) for a in self.state.game_state["current_question"]["attempts"]]) / len(self.state.game_state["current_question"]["attempts"])
            
        # Pass@k is 1 if any of the first k attempts are correct, otherwise 0
        # return 1.0 if any(attempts[:min(self.k, len(attempts))]) else 0.0
        
    def _calculate_best_at_k(self):
        """Best@k returns 1 if any of the k attempts were correct."""
        attempts = self.state.game_state["current_question"]["attempts"]
        if not attempts or len(attempts) < self.k:
            return 0.0
            
        return 1.0 if any(attempts[:self.k]) else 0.0
        
    def _calculate_consensus_at_k(self):
        """
        Cons@k uses majority voting among the k attempts.
        Returns 1 if the majority answer is correct.
        """
        answers = self.state.game_state["current_question"]["answers"]
        if not answers or len(answers) < self.k:
            return 0.0
            
        # Get the k answers we want to consider
        k_answers = answers[:self.k]
        
        # Find the most common answer
        if not k_answers:
            return 0.0
            
        counter = Counter(k_answers)
        most_common_answer = counter.most_common(1)[0][0]
        
        # Check if the most common answer is correct
        return 1.0 if self._check_answer(most_common_answer) else 0.0

    def _update_xatk_metrics(self):
        """Update all x@k metrics when a question is completed."""
        if len(self.state.game_state["current_question"]["attempts"]) >= self.k:
            metrics = self.state.game_state["x@k_metrics"]
            
            # Increment the counter for each metric
            metrics["pass@k"] += self._calculate_pass_at_k()
            metrics["best@k"] += self._calculate_best_at_k()
            metrics["cons@k"] += self._calculate_consensus_at_k()
            metrics["total"] += 1
        
    def _get_xatk_metrics(self):
        """Get all x@k metrics as a dict with average values."""
        metrics = self.state.game_state["x@k_metrics"]
        total = metrics["total"]
        
        if total == 0:
            return {"pass@k": 0.0, "cons@k": 0.0, "best@k": 0.0}
            
        return {
            "pass@k": metrics["pass@k"] / total,
            "cons@k": metrics["cons@k"] / total,
            "best@k": metrics["best@k"] / total
        }
                
    def _calculate_primary_score(self):
        """Calculate the primary score based on the selected evaluation method."""
        if self.eval_method == "accuracy":
            return self._calculate_accuracy()
        elif self.eval_method == "x@k":
            # For x@k, we'll use pass@k as the primary score for backward compatibility
            metrics = self._get_xatk_metrics()
            return metrics["pass@k"]
        
        return 0.0  # Default fallback

    def _show_next_question(self):
        """Display the next question or finish the evaluation if done."""
        # Check if we've collected enough attempts for current question
        current_attempts = len(self.state.game_state["current_question"]["attempts"])
        
        if current_attempts >= self.k or self.question is None:
            # For x@k methods, update all metrics
            if self.eval_method == "x@k":
                self._update_xatk_metrics()
            
            # Reset for the next question
            self.state.game_state["current_question"] = {"attempts": [], "answers": []}
        
            # Check if we've gone through all questions
            if self.questions_seen >= self.total_questions:
                # Calculate final score based on evaluation method
                final_score = self._calculate_primary_score()
                
                # For x@k, include all metrics in the reason
                if self.eval_method == "x@k":
                    metrics = self._get_xatk_metrics()
                    metrics_str = f"pass@{self.k}: {metrics['pass@k']:.3f}, cons@{self.k}: {metrics['cons@k']:.3f}, best@{self.k}: {metrics['best@k']:.3f}"
                    reason = f"Completed all questions with {metrics_str}"
                else:
                    reason = f"Completed all questions with accuracy: {final_score:.3f}"
                    
                self.state.set_custom_game_outcome(player_reward_dict={0: final_score}, reason=reason)
                return False

            try:
                # Get next item from dataset
                item = next(self.dataset)
                # Extract question and answer from the dictionary
                self.question = item['question']
                self.answer = item['answer']
                self.questions_seen += 1
                
                # Add question to state
                self.state.add_observation(from_id=ta.GAME_ID, to_id=self.state.current_player_id, message=str(self.question))
                return True
            except StopIteration:
                # Handle unexpected end of iterator
                self.questions_seen = self.total_questions
                return self._show_next_question()  # Recursively call to handle the end case

    def _check_answer(self, submitted_answer: str) -> bool:
        """
        Check if the submitted answer contains the correct answer.
        """
        # Look for boxed answers in LaTeX
        boxed_pattern = r"\\boxed\{(\d+)\}"
        boxed_matches = re.findall(boxed_pattern, submitted_answer)
        
        # If we found boxed numbers, check those
        if boxed_matches:
            for match in boxed_matches:
                if match.strip() == self.answer.strip():
                    return True
        
        # Check for the number anywhere in the text
        number_pattern = r"\b" + re.escape(self.answer) + r"\b"
        if re.search(number_pattern, submitted_answer):
            return True
            
        # Fall back to the regular math_eval
        try:
            return math_eval(submitted_answer, self.answer)
        except Exception as e:
            # Basic exact match as last resort
            return submitted_answer.strip() == self.answer.strip()

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """Process the agent's answer and update the game state accordingly."""
        # Check if the answer is correct
        correct = self._check_answer(submitted_answer=action)

        # Log the result
        self.state.info["correct"] = correct
        
        # Store the answer and correctness for various metrics
        self.state.game_state["current_question"]["attempts"].append(correct)
        self.state.game_state["current_question"]["answers"].append(action)
        
        # For regular accuracy, update on each answer
        if self.eval_method == "accuracy":
            self.state.game_state["questions_correct"] += int(correct)
            self.state.game_state["questions_total"] += 1
            self.state.info["current_accuracy"] = self._calculate_accuracy()
        
        # For x@k, provide progress info
        elif self.eval_method == "x@k":
            current_attempts = len(self.state.game_state["current_question"]["attempts"])
            self.state.info["attempts_for_current_question"] = f"{current_attempts}/{self.k}"
            
            # If we have ongoing metrics, add them
            if self.state.game_state["x@k_metrics"]["total"] > 0:
                metrics = self._get_xatk_metrics()
                self.state.info["current_pass@k"] = metrics["pass@k"]
                self.state.info["current_cons@k"] = metrics["cons@k"]
                self.state.info["current_best@k"] = metrics["best@k"]
            
        # Get the next question (or keep current if not enough attempts yet)
        self._show_next_question()
        
        return self.state.step()

def choice_answer_clean(pred: str):
    """Clean and standardize multiple choice answers."""
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
    """Parse string representations of numbers, including percentages."""
    num = re.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    """Check if a string can be parsed as a number."""
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    """Convert a string matrix representation to LaTeX pmatrix format."""
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def symbolic_equal(a, b):
    """
    Check if two expressions are symbolically equal using sympy.
    Attempts multiple parsing methods.
    """
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    """Helper function for timeout-based symbolic equality checking."""
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    """Run a function with a timeout to prevent hanging."""
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()


def numeric_equal(prediction: float, reference: float):
    """Compare numerical values with appropriate tolerance."""
    return isclose(reference, prediction, rel_tol=1e-4)


def math_eval(prediction: Union[bool, float, str], reference: Union[float, str], 
             include_percentage: bool = True, is_close: bool = True, 
             timeout: bool = True) -> bool:
    """
    Evaluate if a mathematical prediction is equal to a reference answer.
    
    This function supports various formats including:
    - Exact string matching
    - Multiple choice answers
    - Numerical equality with tolerance
    - Symbolic mathematical equality
    - Matrix and vector equality
    
    Take from: https://github.com/hkust-nlp/simpleRL-reason/blob/main/eval/grader.py

    Which is itself based on:
    - https://github.com/microsoft/ProphetNet/tree/master/CRITIC
    - https://github.com/openai/prm800k
    - https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
    - https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py
    - 
    
    Args:
        prediction: The predicted answer to evaluate
        reference: The ground truth answer to compare against
        include_percentage: Whether to handle percentage values (e.g., 50% = 0.5 = 50)
        is_close: Whether to use approximate numeric equality
        timeout: Whether to apply a timeout for symbolic equality checking
        
    Returns:
        bool: True if the prediction is equal to the reference, False otherwise
    """
    # Handle None values
    if prediction is None or reference is None:
        return False
    
    # Exact string match
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    
    # Multiple choice handling
    if reference in ["A", "B", "C", "D", "E"] and choice_answer_clean(prediction) == reference:
        return True

    try:  # Numerical equality
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # Handle percentage variants
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    # Handle empty predictions
    if not prediction and prediction not in [0, False]:
        return False

    # Prepare for symbolic comparison
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    # Handle pmatrix format
    if "pmatrix" in prediction and not "pmatrix" in reference:
        reference = str_to_pmatrix(reference)

    # Handle basic bracket differences
    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or \
       (prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    
    # Strip braces for simple comparison
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    # Handle arrays/vectors [a, b] vs. [c, d]
    if re.search(r"(\(|\[).+(\)|\])", prediction) is not None and \
       re.search(r"(\(|\[).+(\)|\])", reference) is not None:
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all([math_eval(pred_parts[i], ref_parts[i], include_percentage, is_close)
                   for i in range(len(pred_parts))]):
                return True
    
    # Handle matrices
    if ((prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}")) and
        (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}")) and
        (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}")) and
        (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}"))):
        
        pred_begin = "\\begin{pmatrix}"
        pred_end = "\\end{pmatrix}"
        ref_begin = "\\begin{pmatrix}"
        ref_end = "\\end{pmatrix}"
        
        if "bmatrix" in prediction:
            pred_begin = "\\begin{bmatrix}"
            pred_end = "\\end{bmatrix}"
        if "bmatrix" in reference:
            ref_begin = "\\begin{bmatrix}"
            ref_end = "\\end{bmatrix}"
            
        pred_lines = [line.strip() for line in 
                      prediction[len(pred_begin):-len(pred_end)].split("\\\\") if line.strip()]
        ref_lines = [line.strip() for line in 
                     reference[len(ref_begin):-len(ref_end)].split("\\\\") if line.strip()]
        
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all([math_eval(pred_parts[i], ref_parts[i], include_percentage, is_close)
                               for i in range(len(pred_parts))]):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    # Handle equations with single equal sign
    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (prediction.count("=") == 1 and len(prediction.split("=")[0].strip()) <= 2 and 
          "=" not in reference):
        if math_eval(prediction.split("=")[1], reference, include_percentage, is_close):
            return True
    elif (reference.count("=") == 1 and len(reference.split("=")[0].strip()) <= 2 and 
          "=" not in prediction):
        if math_eval(prediction, reference.split("=")[1], include_percentage, is_close):
            return True

    # Full symbolic equality check with timeout option
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference, timeout=3):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False