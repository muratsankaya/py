import read, copy
from util import *
from logical_classes import *

verbose = 0

class KnowledgeBase(object):
    def __init__(self, facts=[], rules=[]):
        self.facts = facts
        self.rules = rules
        self.ie = InferenceEngine()

    def __repr__(self):
        return 'KnowledgeBase({!r}, {!r})'.format(self.facts, self.rules)

    def __str__(self):
        string = "Knowledge Base: \n"
        string += "\n".join((str(fact) for fact in self.facts)) + "\n"
        string += "\n".join((str(rule) for rule in self.rules))
        return string

    def _get_fact(self, fact):
        """INTERNAL USE ONLY
        Get the fact in the KB that is the same as the fact argument

        Args:
            fact (Fact): Fact we're searching for

        Returns:
            Fact: matching fact
        """
        for kbfact in self.facts:
            if fact == kbfact:
                return kbfact

    def _get_rule(self, rule):
        """INTERNAL USE ONLY
        Get the rule in the KB that is the same as the rule argument

        Args:
            rule (Rule): Rule we're searching for

        Returns:
            Rule: matching rule
        """
        for kbrule in self.rules:
            if rule == kbrule:
                return kbrule

    def kb_add(self, fact_rule):
        """Add a fact or rule to the KB
        Args:
            fact_rule (Fact or Rule) - Fact or Rule to be added
        Returns:
            None
        """
        printv("Adding {!r}", 1, verbose, [fact_rule])
        if isinstance(fact_rule, Fact):
            if fact_rule not in self.facts:
                self.facts.append(fact_rule)
                for rule in self.rules:
                    self.ie.fc_infer(fact_rule, rule, self)
            else:
                if fact_rule.supported_by:
                    ind = self.facts.index(fact_rule)
                    for f in fact_rule.supported_by:
                        self.facts[ind].supported_by.append(f)
                else:
                    ind = self.facts.index(fact_rule)
                    self.facts[ind].asserted = True
        elif isinstance(fact_rule, Rule):
            if fact_rule not in self.rules:
                self.rules.append(fact_rule)
                for fact in self.facts:
                    self.ie.fc_infer(fact, fact_rule, self)
            else:
                if fact_rule.supported_by:
                    ind = self.rules.index(fact_rule)
                    for f in fact_rule.supported_by:
                        self.rules[ind].supported_by.append(f)
                else:
                    ind = self.rules.index(fact_rule)
                    self.rules[ind].asserted = True

    def kb_assert(self, fact_rule):
        """Assert a fact or rule into the KB

        Args:
            fact_rule (Fact or Rule): Fact or Rule we're asserting
        """
        printv("Asserting {!r}", 0, verbose, [fact_rule])
        self.kb_add(fact_rule)

    def kb_ask(self, fact):
        """Ask if a fact is in the KB

        Args:
            fact (Fact) - Statement to be asked (will be converted into a Fact)

        Returns:
            listof Bindings|False - list of Bindings if result found, False otherwise
        """
        print("Asking {!r}".format(fact))
        if factq(fact):
            f = Fact(fact.statement)
            bindings_lst = ListOfBindings()
            # ask matched facts
            for fact in self.facts:
                binding = match(f.statement, fact.statement)
                if binding:
                    bindings_lst.add_bindings(binding, [fact])

            return bindings_lst if bindings_lst.list_of_bindings else []

        else:
            print("Invalid ask:", fact.statement)
            return []

    def kb_retract(self, fact_rule):
        """Retract a fact or a rule from the KB

        Args:
            fact_rule (Fact or Rule) - Fact or Rule to be retracted

        Returns:
            None
        """
        printv("Retracting {!r}", 0, verbose, [fact_rule])

        item = self._get_fact(fact_rule) if factq(fact_rule) else self._get_rule(fact_rule)
        if not item:
            return  # Item not in KB

        if item.asserted:
            if not item.supported_by:
                self._remove_item(item)
            else:
                if factq(item):  # Only facts can be unasserted
                    item.asserted = False
            return

        if not item.supported_by:
            self._remove_item(item)
        
    def _remove_item(self, item):
        if factq(item):
            self.facts.remove(item)
        else:
            self.rules.remove(item)

        # Update supports_facts and supports_rules
        for fact in item.supports_facts:
            self._update_supports(fact, item)
        for rule in item.supports_rules:
            self._update_supports(rule, item)

    def _update_supports(self, item, removed_item):
        item.supported_by = [support for support in item.supported_by if removed_item not in support]

        if not item.supported_by and not item.asserted:
            self._remove_item(item)    

class InferenceEngine(object):
    def fc_infer(self, fact, rule, kb):
        """Forward-chaining to infer new facts and rules"""
        printv('Attempting to infer from {!r} and {!r} => {!r}', 1, verbose,
               [fact.statement, rule.lhs, rule.rhs])

        bindings = match(fact.statement, rule.lhs[0])
        
        if bindings:
            if len(rule.lhs) == 1:
                new_fact_statement = instantiate(rule.rhs, bindings)
                new_fact = Fact(new_fact_statement, supported_by=[[fact, rule]])
                
                existing_fact = kb._get_fact(new_fact)
                if existing_fact:
                    # If it exists, just update the support structures
                    if [fact, rule] not in existing_fact.supported_by:
                        existing_fact.supported_by.append([fact, rule])
                    if new_fact not in fact.supports_facts:
                        fact.supports_facts.append(existing_fact)
                    if new_fact not in rule.supports_facts:
                        rule.supports_facts.append(existing_fact)
                else:
                    kb.kb_add(new_fact)
                    fact.supports_facts.append(new_fact)
                    rule.supports_facts.append(new_fact)
            else:
                new_lhs = [instantiate(statement, bindings) for statement in rule.lhs[1:]]
                new_rhs = instantiate(rule.rhs, bindings)
                new_rule = Rule([new_lhs, new_rhs], supported_by=[[fact, rule]])
                
                existing_rule = kb._get_rule(new_rule)
                if existing_rule:
                    # If it exists, just update the support structures
                    if [fact, rule] not in existing_rule.supported_by:
                        existing_rule.supported_by.append([fact, rule])
                    if new_rule not in fact.supports_rules:
                        fact.supports_rules.append(existing_rule)
                    if new_rule not in rule.supports_rules:
                        rule.supports_rules.append(existing_rule)
                else:
                    kb.kb_add(new_rule)
                    fact.supports_rules.append(new_rule)
                    rule.supports_rules.append(new_rule)
