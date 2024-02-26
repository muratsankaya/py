
def joint_probability(variables, evidence, bn):
    if not variables:
        return 1.0

    var = variables[0]
    remaining_vars = variables[1:]

    if var in evidence:
        node = bn.get_var(var)  
        if node is not None:  
            prob = node.probability(evidence[var], evidence)  
            return prob * joint_probability(remaining_vars, evidence, bn)  
        else:
            return 0  
    else:
        sum_prob = 0
        for value in [True, False]:
            new_evidence = evidence.copy()
            new_evidence[var] = value
            node = bn.get_var(var)
            if node is not None:
                prob_value = node.probability(value, evidence)
                sum_prob += prob_value * joint_probability(remaining_vars, new_evidence, bn)
            else:
                return 0 
        return sum_prob

def ask(var, value, evidence, bn):
    
    # Update the evidence with the hypothesis
    evidence_with_hypothesis = evidence.copy()
    evidence_with_hypothesis[var] = value

    variables = [node.name for node in bn.variables]

    jp = joint_probability(variables, evidence_with_hypothesis, bn)

    alpha = jp + joint_probability(variables, {**evidence, var: not value}, bn)

    # Return P(H|E), the probability of the hypothesis given the evidence
    return jp / alpha if alpha != 0 else 0
