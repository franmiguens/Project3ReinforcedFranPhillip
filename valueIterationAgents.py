# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp
import util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        vals = self.values.copy()
        for i in range(self.iterations):
            for s in self.mdp.getStates():
                qvals = []
                for a in self.mdp.getPossibleActions(s):
                    qval = self.computeQValueFromValues(s, a)
                    qvals.append(qval)
                if qvals:
                    vals[s] = max(qvals)
                else:
                    vals[s] = 0
            self.values = vals.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qval = 0
        for ns, p in self.mdp.getTransitionStatesAndProbs(state, action):
            qval += p*(self.mdp.getReward(state, action, ns) +
                       (self.discount*self.values[ns]))
        return qval
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if (self.mdp.isTerminal(state)):
            return None
        policy = {}

        qvals = []
        acts = []

        for a in self.mdp.getPossibleActions(state):
            qval = self.computeQValueFromValues(state, a)
            qvals.append(qval)
            acts.append(a)
        if not qvals:
            policy[state] = (0, None)
        else:
            policy[state] = (max(qvals), acts[qvals.index(max(qvals))])
        return policy[state][1]
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        counter = 0
        s = self.mdp.getStates()
        length = len(s)
        while counter < self.iterations:
            qvals = []
            for a in self.mdp.getPossibleActions(s[counter % length]):
                qval = self.computeQValueFromValues(s[counter % length], a)
                qvals.append(qval)
            if qvals:
                self.values[s[counter % length]] = max(qvals)
            else:
                self.values[s[counter % length]] = 0
            counter += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = {}
        for i in self.mdp.getStates():
            for j in self.mdp.getPossibleActions(i):
                for k in self.mdp.getTransitionStatesAndProbs(i, j):
                    if k[0] not in predecessors:
                        predecessors[k[0]] = {i}
                    else:
                        predecessors[k[0]].add(i)
        pq = util.PriorityQueue()
        vals = util.Counter()
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                qvals = []
                for a in self.mdp.getPossibleActions(s):
                    qvals.append(self.computeQValueFromValues(s, a))
                if qvals:
                    diff = abs(max(qvals)-self.values[s])
                    pq.update(s, -diff)
                    vals[s] = max(qvals)
        for i in range(self.iterations):
            if not pq.isEmpty():
                state = pq.pop()
                if not self.mdp.isTerminal(state):
                    self.values[state] = vals[state]
                for pred in predecessors[state]:
                    if not self.mdp.isTerminal(pred):
                        qvals = []
                        for a in self.mdp.getPossibleActions(pred):
                            qvals.append(self.computeQValueFromValues(pred, a))
                        if qvals:
                            diff = abs(max(qvals)-self.values[pred])
                            if diff > self.theta:
                                pq.update(pred, -diff)
                                vals[pred] = max(qvals)
        "*** YOUR CODE HERE ***"
