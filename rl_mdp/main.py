from util import create_mdp, create_policy_1, create_policy_2
from model_free_prediction.monte_carlo_evaluator import MCEvaluator
from model_free_prediction.td_evaluator import TDEvaluator
from model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator
import numpy as np

def compare_policies(v_pi1: np.ndarray, v_pi2: np.ndarray) -> None:
    """
    Compares two policies based on their value functions state by state.
    
    :param v_pi1: Value function for policy 1.
    :param v_pi2: Value function for policy 2.
    """
    for i, (val1, val2) in enumerate(zip(v_pi1, v_pi2)):
        if val1 > val2:
            print(f"State {i}: Policy 1 is better (Value: {val1:.4f} vs {val2:.4f})")
        elif val1 < val2:
            print(f"State {i}: Policy 2 is better (Value: {val1:.4f} vs {val2:.4f})")
        else:
            print(f"State {i}: Both policies are equally good (Value: {val1:.4f})")


def main() -> None:
    # Assume create_mdp, create_policy_1, create_policy_2 are defined elsewhere
    mdp = create_mdp()
    policy_1 = create_policy_1()
    policy_2 = create_policy_2()

    # Initialize the evaluators
    mc_evaluator = MCEvaluator(mdp)
    td_evaluator = TDEvaluator(mdp, alpha=0.1)
    tdlambda_evaluator = TDLambdaEvaluator(mdp, alpha=0.1, lambd=0.5)

    # Evaluate both policies using Monte Carlo
    print("Evaluating using Monte Carlo:")
    v_pi1_mc = mc_evaluator.evaluate(policy_1, num_episodes=1000)
    v_pi2_mc = mc_evaluator.evaluate(policy_2, num_episodes=1000)
    compare_policies(v_pi1_mc, v_pi2_mc)

    # Evaluate both policies using TD(0)
    print("\nEvaluating using TD(0):")
    v_pi1_td = td_evaluator.evaluate(policy_1, num_episodes=1000)
    v_pi2_td = td_evaluator.evaluate(policy_2, num_episodes=1000)
    compare_policies(v_pi1_td, v_pi2_td)

    # Evaluate both policies using TD(lambda)
    print("\nEvaluating using TD(lambda):")
    v_pi1_tdlambda = tdlambda_evaluator.evaluate(policy_1, num_episodes=1000)
    v_pi2_tdlambda = tdlambda_evaluator.evaluate(policy_2, num_episodes=1000)
    compare_policies(v_pi1_tdlambda, v_pi2_tdlambda)

if __name__ == "__main__":
    main()

