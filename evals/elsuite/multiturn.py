from pyparsing import Any
import evals
from evals.api import CompletionFn
from evals.eval import Eval
import evals.metrics


class MultiTurnEval(Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        *args,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)

    def eval_sample(self, sample: Any, *_):
        debate_prompt = sample[0]
        expected_answer = sample[1]

        roles = ["Pro", "Con"]
        assert len(self.completion_fns) == len(roles), "Must have one completion_fn per role"

        n_turns = 3
        messages = [
            f"Debate prompt: {debate_prompt}\n\n",
        ]
        for i in range(n_turns):
            messages.append(f"Turn #{i} ---------------")
            for role, completion_fn in zip(roles, self.completion_fns):
                msg_history = "\n".join(messages)
                prompt = f"{msg_history}\n\n{role}, please give a brief response to the above argument. (Remaining turns: {n_turns - i})\n{role} response:"
                answer = completion_fn(prompt).get_completions()[0]
                new_msg = f"{role} response: {answer}"
                messages.append(new_msg)
        
        print("--------------------------------------------")

        transcript = "\n".join(messages)
        print(transcript)
        
        # TODO: Do some evaluation here, e.g. judge the transcript
        # Dummy evaluation for now
        sampled_answer = "Pro"
        evals.record_and_check_match(transcript, sampled_answer, expected=expected_answer)

    def run(self, recorder):
        samples = [
            ("Be it resolved, AI research and development poses an existential threat.", "Pro"),
        ]
        self.eval_all_samples(recorder, samples)
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }