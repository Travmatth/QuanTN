from quantn import Qubit

def test_qubit_init():
  q_0 = Qubit()
  print(q_0.shape)
  assert q_0.shape[0] == 2

def test_qubit_0ket_bitstring():
  q_0 = Qubit()
  for i in range(100):
    assert q_0.bitstring() == "0"

