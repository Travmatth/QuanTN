import random
import numpy as np
import tensornetwork
from typing import Optional, Text, Tuple
from quantn.qubit import Qubit
from tensornetwork.network_components import Tensor, Node, BaseNode, Edge, \
    connect, contract

class XGate(Qubit):
  def __init__(self, target: Qubit):
    xgate = Node(np.array([[0, 1], [1, 0]]))
    xgate[0] ^ target["edge"]
    qubit = xgate @ target
    super().__init__(tensor=qubit.tensor)

