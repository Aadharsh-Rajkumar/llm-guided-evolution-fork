# Reducing gate cost without breaking a Qiskit circuit

Circuit size is one of the objectives, and smaller is better. It is a WEIGHTED gate
count of the gates in the readout qubit's backward light cone (the only gates that
can change the prediction): every single-qubit gate costs 1 and every two-qubit gate
(cx, cz, swap, rzz, crx, ...) costs 5, because two-qubit gates are about ten times
noisier on hardware. Barriers and measurements cost 0. An edit that keeps accuracy
and removes gates is an improvement; an edit that adds gates must buy accuracy.
Ways to lower the cost while keeping the circuit valid:
- Fewer layers (`N_LAYERS`) remove a full rotation + entangling block each.
- Use a linear chain instead of a ring (saves one two-qubit gate per layer).
- Entangle only the pairs that feed the readout qubit; gates outside its light cone
  cannot change the prediction.
- Drop rotations that add nothing: an RZ applied directly to |0> only adds a phase,
  and a diagonal gate (RZ, CZ) right before readout cannot change the probabilities.
- Merge rotations: RY(a) followed by RY(b) on the same qubit equals RY(a + b).
- Use one rotation axis per qubit per layer instead of RY + RZ pairs.
If you remove gates that consumed trainable parameters, also shrink the
`ParameterVector` size in the circuit-assembly function or keep offsets consistent,
so every weight in the returned list is actually used by the circuit.
