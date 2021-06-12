from collections import Counter
import math


def entropy(probs):
    """
    function to calculate the entropy of probability of observations -p*log2*p
    :param probs:
    :return:
    """
    return sum([-prob*math.log(prob, 2) for prob in probs])


def entropy_of_list(a_list):
    """
    function to calculate the entropy of the given datasets/list with respect to target attributes
    :param a_list:
    :return:
    """
    print("A-list", a_list)
    cnt = Counter(x for x in a_list)  # counter calculates the proportion of a class
    print("\nClasses: ", cnt)
    print("No and Yes Classes: ", a_list.name, cnt)
    num_instances = len(a_list) * 1.0
    print("\n Number of instances of the Current Subclass is {0}: ".format(num_instances))
    probs = [x / num_instances for x in cnt.values()]  # x means no of YES/NO
    print("\n Classes: ", min(cnt), max(cnt))
    print("\n Probabilities of class{0} is {1}:".format(min(cnt), min(probs)))
    print("\n Probabilities of class{0} is {1}:".format(max(cnt), max(probs)))
    return entropy(probs)  # call entropy


def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    """
    takes a dataframe of attributes and quantifies the entropy of a target attribute after performing a split along the
    values of another attribute.
    :param df:
    :param split_attribute_name:
    :param target_attribute_name:
    :param trace:
    """
    print("information Gain Calculations of ", split_attribute_name)
    # split data by possible vals of attribute:
    df_split = df.groupby(split_attribute_name)
    for name,group in df_split:
        print("Name:\n", name)
        print("Group:\n", group)
    # calculate Entropy for Target Attribute, as well as proportion of obs in each data-split
    nobs = len(df.index) * 1.0
    print("NOBS", nobs)
    df_agg_ent = df_split.agg({target_attribute_name : [entropy_of_list, lambda x: len(x)/nobs]})[target_attribute_name]
    print([target_attribute_name])
    print("Entropy List", entropy_of_list)
    print("DFAGGENT", df_agg_ent)
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    if trace:  # helps understand what fxn is doing:
        print(df_agg_ent)

    # Calculate information gain:
    new_entropy = sum(df_agg_ent['Entropy'] * df_agg_ent['PropObservations'])
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - new_entropy