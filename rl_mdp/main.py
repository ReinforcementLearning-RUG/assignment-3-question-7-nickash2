from util import create_mdp, create_policy_1, create_policy_2
from model_free_prediction.monte_carlo_evaluator import MCEvaluator
from model_free_prediction.td_evaluator import TDEvaluator
from model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator
import numpy as np

def compare_policies_overall(v_pi1: np.ndarray, v_pi2: np.ndarray) -> None:
    """
    Compares two policies based on their overall value functions by summing the values across all states.
    
    :param v_pi1: Value function for policy 1.
    :param v_pi2: Value function for policy 2.
    """
    total_v_pi1 = np.sum(v_pi1)
    total_v_pi2 = np.sum(v_pi2)

    # Output results
    print(f"Sum of Value Functions: Policy 1 = {total_v_pi1:.4f}, Policy 2 = {total_v_pi2:.4f}")

    # Determine overall better policy
    if total_v_pi1 > total_v_pi2:
        print("Overall, Policy 1 is better based on the sum of value functions.")
    elif total_v_pi1 < total_v_pi2:
        print("Overall, Policy 2 is better based on the sum of value functions.")
    else:
        print("Both policies are equally good based on the sum of value functions.")


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
    compare_policies_overall(v_pi1_mc, v_pi2_mc)

    # Evaluate both policies using TD(0)
    print("\nEvaluating using TD(0):")
    v_pi1_td = td_evaluator.evaluate(policy_1, num_episodes=1000)
    v_pi2_td = td_evaluator.evaluate(policy_2, num_episodes=1000)
    compare_policies_overall(v_pi1_td, v_pi2_td)

    # Evaluate both policies using TD(lambda)
    print("\nEvaluating using TD(lambda):")
    v_pi1_tdlambda = tdlambda_evaluator.evaluate(policy_1, num_episodes=1000)
    v_pi2_tdlambda = tdlambda_evaluator.evaluate(policy_2, num_episodes=1000)
    compare_policies_overall(v_pi1_tdlambda, v_pi2_tdlambda)

if __name__ == "__main__":
    main()

